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
            "predict_labels": [self._model.vocab.get_token_from_index(index, namespace="labels")
                               for index in output_dict["predicted_tags"]],
            "sample_id": output_dict["meta"]["sample_id"],
            "start": output_dict["meta"]["start"],
            "end": output_dict["meta"]["end"],
        }
        if "true_labels" in inputs:
            outputs["true_labels"] = inputs["true_labels"]
        return outputs

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = [ch for ch in json_dict["sentence"]]
        instance = self._dataset_reader.text_to_instance(tokens,
            json_dict["sample_id"], json_dict["start"], json_dict["end"])
        return instance