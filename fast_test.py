import os
import tqdm
import argparse
import collections
from typing import Any, Union, Dict, Iterable, List, Optional, Tuple

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from models import BertSlotTagging
from predictors import SlotFillingPredictor
from dataset_readers import MultiFileDatasetReader, parse_ann, parse_seq_out
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedBertIndexer

from allennlp.data import vocabulary
vocabulary.DEFAULT_OOV_TOKEN = "[UNK]"  # set for bert


def main(args):
    for output_dir in args.output_dir:
        test_one_model(output_dir, args.test_data_dir, args.batch_size)


def test_one_model(output_dir, test_data_dir, batch_size=32):
    anns_dir = os.path.join(output_dir, "anns")
    if not os.path.exists(anns_dir):
        os.mkdir(anns_dir)
    archive = load_archive(output_dir, cuda_device=1)
    predictor = Predictor.from_archive(archive=archive, predictor_name="bert_st")
    for filename in tqdm.tqdm([item for item in os.listdir(test_data_dir) if item[-4:] == ".txt"]):
        sample_id = filename[:-4]
        input_jsons = []
        output_jsons = []
        for item in parse_ann(os.path.join(test_data_dir, filename)):
            input_jsons.append({"sentence": item["sentence"], "sample_id": int(sample_id), "start": item["start"], "end": item["end"]})
        for start_idx in range(0, len(input_jsons), batch_size):
            output_jsons.extend(predictor.predict_list(input_jsons[start_idx:start_idx+batch_size]))
        one_sample_set = []
        for output_json in output_jsons:
            predicted_tags = output_json["predict_labels"]
            for start_idx, end_idx, label in parse_seq_out(predicted_tags):
                one_sample_set.append((output_json["start"] + start_idx, output_json["start"] + end_idx + 1, label, output_json["sentence"][start_idx:end_idx+1]))
        with open(os.path.join(anns_dir, f"{sample_id}.ann"), "w") as f:
            for idx, (start, end, label, value) in enumerate(one_sample_set):
                f.write(f"T{idx+1}\t{label} {start} {end}\t{value}\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--output_dir", type=str, nargs="+", required=True,
                            help="the directory that stores training output")
    arg_parser.add_argument("--test_data_dir", type=str, default="./data/tianchi/chusai_xuanshou")
    arg_parser.add_argument("--batch_size", type=int, default=16)
    args = arg_parser.parse_args()
    main(args)