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
    for filename in tqdm.tqdm([item for item in os.listdir(args.test_data_dir) if item[-4:] == ".txt"]):
        span_rate = collections.defaultdict(int)
        sample_id = filename[:-4]
        for output_dir in args.output_dir:
            ann_file = os.path.join(output_dir, "anns", f"{sample_id}.ann")
            with open(ann_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    _, tmp, value = line.strip().split("\t")
                    label, start, end = tmp.split(" ")
                    span_rate[(label, start, end, value)] += 1
        with open(os.path.join(args.test_data_dir, f"{sample_id}.ann"), "w") as f:
            idx = 0
            for (label, start, end, value), count in span_rate.items():
                if count >= args.ensemble_threshold:
                    f.write(f"T{idx+1}\t{label} {start} {end}\t{value}\n")
                    idx += 1


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--output_dir", type=str, nargs="+", required=True,
                            help="the directory that stores training output")
    arg_parser.add_argument("--test_data_dir", type=str, default="./data/tianchi/chusai_xuanshou")
    arg_parser.add_argument("--ensemble_threshold", type=int, default=1)
    args = arg_parser.parse_args()
    main(args)