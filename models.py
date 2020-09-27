from typing import Dict, Optional, List, Any

import os
import torch
import logging
import collections

from allennlp.common import Params
from allennlp.models import Model
from allennlp.modules import TimeDistributed
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from pytorch_pretrained_bert import BertTokenizer

logger = logging.getLogger(__name__)


@Model.register("bert_st")
class BertSlotTagging(Model):
    
    def __init__(self, 
                 vocab: Vocabulary,
                 bert_embedder: Optional[PretrainedBertEmbedder] = None,
                 encoder: Optional[Seq2SeqEncoder] = None,
                 dropout: Optional[float] = None,
                 use_crf: bool = True,
                 add_random_noise: bool = False,
                 add_attack_noise: bool = False,
                 noise_norm: Optional[float] = None,
                 noise_loss_prob: Optional[float] = None,
                 add_noise_for: str = "slot_token",
                 training_ann_dir: str = None,
                 matching_longer_than: int = None) -> None:
        super().__init__(vocab)

        if bert_embedder:
            self.use_bert = True
            self.bert_embedder = bert_embedder
        else:
            self.use_bert = False
            self.basic_embedder = BasicTextFieldEmbedder({
                "tokens": Embedding(vocab.get_vocab_size(namespace="tokens"), 1024)
            })
            self.rnn = Seq2SeqEncoder.from_params(Params({     
                "type": "lstm",
                "input_size": 1024,
                "hidden_size": 512,
                "bidirectional": True,
                "batch_first": True
            }))

        self.encoder = encoder

        if encoder:
            hidden2tag_in_dim = encoder.get_output_dim()
        else:
            hidden2tag_in_dim = bert_embedder.get_output_dim()
        self.hidden2tag = TimeDistributed(torch.nn.Linear(
            in_features=hidden2tag_in_dim,
            out_features=vocab.get_vocab_size("labels")))
        
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        
        self.use_crf = use_crf
        if use_crf:
            crf_constraints = allowed_transitions(
                constraint_type="BIO",
                labels=vocab.get_index_to_token_vocabulary("labels")
            )
            self.crf = ConditionalRandomField(
                num_tags=vocab.get_vocab_size("labels"),
                constraints=crf_constraints,
                include_start_end_transitions=True
            )
        
        self.f1 = SpanBasedF1Measure(vocab, 
                                     tag_namespace="labels",
                                     ignore_classes=["news/type","negation",
                                                     "demonstrative_reference",
                                                     "timer/noun","timer/attributes"],
                                     label_encoding="BIO")
        
        self.add_random_noise = add_random_noise
        self.add_attack_noise = add_attack_noise
        assert not (add_random_noise and add_attack_noise), "both random and attack noise applied"
        if add_random_noise or add_attack_noise:
            assert noise_norm is not None
            assert noise_loss_prob is not None and 0. <= noise_loss_prob <= 1.
            self.noise_norm = noise_norm
            self.noise_loss_prob = noise_loss_prob
            assert add_noise_for in ["slot_token", "context_token", "all_token"]
            self.add_noise_for = add_noise_for
        
        if training_ann_dir is not None:
            self.use_matching = True
            self.label_dicts = collections.defaultdict(set)
            self.token_dicts = collections.defaultdict(set)
            for filename in os.listdir(training_ann_dir):
                if filename.endswith(".ann"):
                    for line in open(os.path.join(training_ann_dir, filename), "r"):
                        _, tmp, token = line.strip().split("\t")
                        label, _, _ = tmp.split(" ")
                        if matching_longer_than is not None and len(token) <= matching_longer_than:
                            continue
                        self.label_dicts[label].add(token)
                        self.token_dicts[token].add(label)
            for token, labels in self.token_dicts.items():
                if len(labels) >= 2:
                    logger.info(f"Warning: token {token} has multiple labels: {list(labels)}")
                    for label in labels:
                        self.label_dicts[label].remove(token)
            logger.info("Loaded dictionary for each slot type.")
            for k, v in self.label_dicts.items():
                logger.info(f"{k} - total {len(v)} - {list(v)}")
        else:
            self.use_matching = False

    def _inner_forward(self,
                       embeddings: torch.Tensor,
                       mask: torch.Tensor,
                       slot_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward from embedding space to a loss or predicted-tags.
        """
        output = {}

        if self.encoder:
            encoder_out = self.encoder(embeddings, mask)
            if self.dropout:
                encoder_out = self.dropout(encoder_out)
            output["encoder_out"] = encoder_out
        else:
            encoder_out = embeddings
        
        tag_logits = self.hidden2tag(encoder_out)
        output["tag_logits"] = tag_logits

        if self.use_crf:
            best_paths = self.crf.viterbi_tags(tag_logits, mask)
            predicted_tags = [x for x, y in best_paths]  # get the tags and ignore the score
            if self.use_matching:
                self.matching(predicted_tags)
            predicted_score = [y for _, y in best_paths]
            output["predicted_tags"] = predicted_tags
            output["predicted_score"] = predicted_score
        else:
            output["predicted_tags"] = torch.argmax(tag_logits, dim=-1)  # pylint: disable=no-member
        
        if slot_labels is not None:
            if self.use_crf:
                log_likelihood = self.crf(tag_logits, slot_labels, mask)  # returns log-likelihood
                output["loss"] = -1.0 * log_likelihood  # add negative log-likelihood as loss
                
                # Represent viterbi tags as "class probabilities" that we can
                # feed into the metrics
                class_probabilities = tag_logits * 0.
                for i, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        class_probabilities[i, j, tag_id] = 1
                self.f1(class_probabilities, slot_labels, mask.float())
            else:
                output["loss"] = sequence_cross_entropy_with_logits(tag_logits, slot_labels, mask)
                self.f1(tag_logits, slot_labels, mask.float())
        
        return output
    
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                meta: List[Any],
                slot_labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Return a Dict (str -> torch.Tensor), which contains fields:
            mask - the mask matrix of ``sentence``, shape: (batch_size, seq_length)
            embeddings - the embedded tokens, shape: (batch_size, seq_length, embed_size)
            encoder_out - the output of contextual encoder, shape: (batch_size, seq_length, num_features)
            tag_logits - the output of tag projection layer, shape: (batch_size, seq_length, num_tags)
            predicted_tags - the output of CRF layer (use viterbi algorithm to obtain best paths),
                             shape: (batch_size, seq_length)
        """
        # print("bert(token piece ids) shape:", sentence["bert"].shape, sentence["bert"][3])
        # print("bert-offsets shape:", sentence["bert-offsets"].shape, sentence["bert-offsets"][3])
        # print("bert-type-ids shape:", sentence["bert-type-ids"].shape, sentence["bert-type-ids"][3])
        # print("slot-labels shape:", slot_labels.shape, slot_labels[3])
        # bert_tokenizer = BertTokenizer.from_pretrained("/home/yym2019/downloads/word-embeddings/bert-base-chinese/vocab.txt")
        # print("bert wordpieces:", bert_tokenizer.convert_ids_to_tokens([tensor.item() for tensor in sentence["bert"][3]]))
        # exit()
        self.meta = meta
        output = {"meta": meta}

        mask = get_text_field_mask(sentence)
        output["mask"] = mask
        # print("mask shape:", mask.shape)
        
        if self.use_bert:
            embeddings = self.bert_embedder(sentence["bert"], sentence["bert-offsets"], sentence["bert-type-ids"])
            if self.dropout:
                embeddings = self.dropout(embeddings)
            output["embeddings"] = embeddings
            # print("embeddings shape:", embeddings.shape)
        else:
            embeddings = self.basic_embedder(sentence)
            if self.dropout:
                embeddings = self.dropout(embeddings)
            output["embeddings"] = embeddings
            embeddings = self.rnn(embeddings, mask)
            if self.dropout:
                embeddings = self.dropout(embeddings)
            output["rnn_out"] = embeddings
        
        if not self.training:  # when predict or evaluate, no need for adding noise
            output.update(self._inner_forward(embeddings, mask, slot_labels))
        elif not self.add_random_noise and not self.add_attack_noise:  # for baseline
            output.update(self._inner_forward(embeddings, mask, slot_labels))
        else:  # add random noise or attack noise for open-vocabulary slots
            if self.add_random_noise:  # add random noise
                unnormalized_noise = torch.randn(embeddings.shape).to(device=embeddings.device)
            else:  # add attack noise
                normal_loss = self._inner_forward(embeddings, mask, slot_labels)["loss"]
                embeddings.retain_grad()
                normal_loss.backward(retain_graph=True)
                unnormalized_noise = embeddings.grad.detach_()
                for p in self.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
            norm = unnormalized_noise.norm(p=2, dim=-1)
            normalized_noise = unnormalized_noise / (norm.unsqueeze(dim=-1) + 1e-10)  # add 1e-10 to avoid NaN
            if self.add_noise_for == "slot_token":
                noise_mask = (slot_labels == 0)
            elif self.add_noise_for == "context_token":
                noise_mask = (slot_labels != 0)
            elif self.add_noise_for == "all_token":
                noise_mask = torch.zeros_like(slot_labels)
            ov_slot_noise = self.noise_norm * normalized_noise * noise_mask.unsqueeze(dim=-1).float()
            output["ov_slot_noise"] = ov_slot_noise
            noise_embeddings = embeddings + ov_slot_noise
            normal_sample_loss = self._inner_forward(embeddings, mask, slot_labels)["loss"]
            noise_sample_loss = self._inner_forward(noise_embeddings, mask, slot_labels)["loss"]
            loss = normal_sample_loss * (1 - self.noise_loss_prob) + noise_sample_loss * self.noise_loss_prob
            output["loss"] = loss
        return output
    
    def matching(self, predited_tags):
        sentences = [each["sentence"] for each in self.meta]
        for sentence, tags in zip(sentences, predited_tags):
            for slot_type, slot_dict in self.label_dicts.items():
                for token in slot_dict:
                    start_idx = 0
                    find_idx = sentence.find(token, start_idx)
                    while find_idx != -1:
                        o_tag_idx = self.vocab.get_token_index("O", "labels")
                        if all(tags[idx] == o_tag_idx for idx in range(find_idx, find_idx + len(token))):
                            tags[find_idx] = self.vocab.get_token_index(f"B-{slot_type}", "labels")
                            for i_idx in range(find_idx + 1, find_idx + len(token)):
                                tags[i_idx] = self.vocab.get_token_index(f"I-{slot_type}", "labels")
                        start_idx = find_idx + len(token)
                        find_idx = sentence.find(token, start_idx)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        matric = self.f1.get_metric(reset)
        return {"precision": matric["precision-overall"],
                "recall": matric["recall-overall"],
                "f1": matric["f1-measure-overall"]}