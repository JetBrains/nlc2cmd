import argparse as ag
from pathlib import Path
import json
from copy import copy
from typing import List, Union, Dict
from nlpaug import Augmenter
from tqdm import tqdm

from parse_config import build_augmentation_pipeline, read_config
from utils import find_free_file


def parse_args():
    parser = ag.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path)
    parser.add_argument('-i', '--input-json', type=Path)
    parser.add_argument('-o', '--output-json', type=Path)
    parser.add_argument('-t', '--text_key', type=Path)
    return parser.parse_args()


def read_original_data(path: Union[Path, str]):
    path = Path(path)
    with path.open('r') as istream:
        data = json.load(istream)
    result = []
    for k, v in data.items():
        v.update({'id': k, 'original': True})
        result.append(v)
    return result


def augment_dataset(
    augmenter: Augmenter,
    data: List[Dict],
    text_key: str,
    out_file: Path,
    aug_config: Dict,
    original_key: str='original'
):
    with find_free_file(out_file).open("x") as ostream:
        # Dump config so it will be easy to know how data was augmented
        ostream.write(f"{json.dumps(aug_config)}\n")
        for entry in tqdm(data):
            # Save original example:
            ostream.write(f"{json.dumps(entry)}\n")
            example = entry[text_key]
            variants = augmenter.augment(example)
            if isinstance(variants, str):
                variants = [variants]
            for var in variants:
                new_entry = copy(entry)
                new_entry.update({
                    text_key: var,
                    original_key: False
                })
                ostream.write(f"{json.dumps(new_entry)}\n")


if __name__ == '__main__':
    args = parse_args()
    pipeline = build_augmentation_pipeline(args.config)
    config = read_config(args.config)
    data = read_original_data(args.input_json)
    augment_dataset(
        augmenter=pipeline,
        data=data,
        text_key=args.text_key,
        out_file=args.output_json,
        aug_config=config
    )
