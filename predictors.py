import numpy as np
from typing import Tuple
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor

@Predictor.register("bert_st")
class SlotFillingPredictor(Predictor):

    def predict(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        outputs = {
            "tokens": [ch for ch in inputs["sentence"]],
            "predict_labels": [self._model.vocab.get_token_from_index(index, namespace="slot_labels")
                               for index in output_dict["predicted_tags"]],
            "predict_intent": self._model.vocab.get_token_from_index(output_dict["predicted_intent"], namespace="intent_labels")
        }
        if "true_labels" in inputs:
            outputs["true_labels"] = inputs["true_labels"]
        return outputs

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = [ch for ch in json_dict["sentence"]]
        instance = self._dataset_reader.text_to_instance(tokens=tokens)
        return instance