from typing import Optional, Iterator, List, Dict, Tuple

import os
import json
import logging
import random

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary


logger = logging.getLogger(__name__)


def parse_seq_out(seq_out: List[str]) -> List[Tuple[int, int, str]]:
    results = []
    last_slot_type, start_idx = None, None
    for idx, label in enumerate(seq_out):
        if label == "O":
            if last_slot_type is not None:
                results.append((start_idx, idx - 1, last_slot_type))
                last_slot_type = None
                start_idx = None
            else:
                continue
        else:
            if last_slot_type is not None:
                assert label[0] == "B" or last_slot_type == label[2:]
                if label[0] == "I":
                    continue
                else:
                    results.append((start_idx, idx - 1, last_slot_type))
                    last_slot_type = label[2:]
                    start_idx = idx
            else:
                assert label[0] == "B"
                last_slot_type = label[2:]
                start_idx = idx
    if last_slot_type is not None:
        results.append((start_idx, len(seq_out) - 1, last_slot_type))
    return results


def parse_ann(txt_file, ann_file=None):
    """
    接收txt_file和ann_file文件路径，返回一个列表，每个元素包含一个短句样本。
    短句样本为句子和BIO标注的元组。
    """
    with open(txt_file, 'r') as f:
        input_line = f.readline()
        input_line = input_line.rstrip()

    if ann_file is not None:
        with open(ann_file, 'r') as f:
            ann_lines = f.readlines()
        all_tags = ["O" for _ in range(len(input_line))]
        for ann_line in ann_lines:
            assert ann_line.strip()
            _, ano, slot_value = ann_line.strip().split("\t")
            tag, start, end = ano.split()
            start, end = int(start), int(end)
            assert input_line[start:end] == slot_value, (input_line[start:end], slot_value)
            all_tags[start] = f"B-{tag}"
            for idx in range(start + 1, end):
                all_tags[idx] = f"I-{tag}"
    
    sentences = input_line.split()
    new_sentences = []
    for sentence in sentences:
        if len(sentence) > 500:
            for sub_sent in sentence.split("。"):
                if sub_sent.strip():
                    new_sentences.append(sub_sent.strip())
        else:
            new_sentences.append(sentence)
    sentences = new_sentences
    # assert all(len(s) > 4 for s in sentences), sentences
    sentence_spans = []
    start_from = 0
    for sentence in sentences:
        start_idx = input_line.find(sentence, start_from)
        assert start_idx != -1
        end_idx = start_idx + len(sentence)
        start_from = end_idx
        obj = {
            "sentence": sentence,
            "start": start_idx,
            "end": end_idx
        }
        if ann_file is not None:
            obj["slots"] = all_tags[start_idx:end_idx]
        sentence_spans.append(obj)
    # [{'sentence': '补气养血、调经止带，用于月经不调、经期腹痛', 'start': 1, 'end': 22}, {'sentence': '非处方药物（甲类）,国家基本药物目录（2012）', 'start': 24, 'end': 48}]
    # print(sentence_spans[:10])
    # print(f"longest span: {max(len(span['sentence']) for span in sentence_spans)}")
    return sentence_spans


@DatasetReader.register("multi_file")
class MultiFileDatasetReader(DatasetReader):
    def __init__(self,
                 data_path: str = "./data/tianchi/train",
                 token_indexers: Dict[str, TokenIndexer] = None,
                 random_drop: Optional[float] = None) -> None:
        super().__init__(lazy=False)
        self.data_path = data_path
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.random_drop = random_drop
    
    def text_to_instance(self, tokens: List[str], sample_id: int, start: int,
                         end: int, slots: List[str] = None) -> Instance:
        sentence_field = TextField([Token(token) for token in tokens], self.token_indexers)
        meta_field = MetadataField({"sentence": "".join(tokens), "sample_id": sample_id, "start": start, "end": end})
        fields = {"sentence": sentence_field, "meta": meta_field}
        if slots:
            slot_label_field = SequenceLabelField(labels=slots, sequence_field=sentence_field)
            fields["slot_labels"] = slot_label_field
        
        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, 'r') as f_indexes:
            sample_indexes = [int(line.strip()) for line in f_indexes.readlines() if line.strip()]
        if len(sample_indexes) > 600:
            train = True
        else:
            train = False
        for sample_id in sample_indexes:
            sentence_spans = parse_ann(os.path.join(self.data_path, f"{sample_id}.txt"),
                                       os.path.join(self.data_path, f"{sample_id}.ann"))
            for item in sentence_spans:
                sentence = item["sentence"]
                slots = item["slots"]
                start = item["start"]
                end = item["end"]
                assert sentence.strip() and slots
                assert len(sentence) == len(slots)
                tokens: List[str] = [ch for ch in sentence]
                if len(tokens) > 500:  # TODO
                    logger.info(f"Dropped sample `{sentence[:50]}...` since the length > 500.")
                    continue
                if train and all(slot == "O" for slot in slots) and self.random_drop is not None:
                    if len(slots) < 10:
                        logger.info(f"Dropped sample `{sentence}` since the length < 10.")
                        continue
                    if random.random() < self.random_drop:
                        logger.info(f"Randomly dropped sample `{sentence}` as it hits prob {self.random_drop}")
                        continue
                yield self.text_to_instance(tokens, sample_id, start, end, slots)


if __name__ == "__main__":
    parse_ann("./data/tianchi/train/1.txt", "./data/tianchi/train/1.ann")
    reader = MultiFileDatasetReader()
    instances = reader.read("./data/tianchi/train.txt")
    print(instances[0])
    print(instances[0]["meta"].metadata)