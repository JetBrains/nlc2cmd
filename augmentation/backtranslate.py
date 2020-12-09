import argparse as ag
from pathlib import Path
import json
from typing import List, Dict, Generator, Iterable, Union
from copy import copy
from nlpaug import Augmenter
from nlpaug.flow import Sequential
from nlpaug.augmenter.word import BackTranslationAug
import json
from tqdm import tqdm


def parse_args():
    parser = ag.ArgumentParser()
    parser.add_argument("-i", "--input-json", type=Path)
    parser.add_argument("-c", "--chain", type=str)
    parser.add_argument("-o", "--output-json", type=Path)
    parser.add_argument("-d", "--device", type=str)
    parser.add_argument("-t", "--text-key", type=str)
    parser.add_argument("-b", "--batch-size", type=int)
    args = parser.parse_args()
    args.chain = parse_chain(args.chain)
    return args


def parse_chain(chain):
    return chain.split("-")


def find_free_file(path: Union[str, Path]) -> Path:
    path = Path(path) if isinstance(path, str) else path
    new_path = path
    i = 2
    while new_path.exists():
        new_path = path.parent / f"{path.stem}_{i}{path.suffix}"
        i += 1
    return new_path


def augment_dataset(
    augmenter: Augmenter,
    data: List[Dict],
    text_key: str,
    out_file: Path,
    batch_size: int,
    is_original_key: str = 'original'
) -> Generator[Dict, None, None]:
    with find_free_file(out_file).open("x") as ostream:
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i+batch_size]
            examples = [e[text_key] for e in batch]
            variants = augmenter.augment(examples)
            for entry, var in zip(batch, variants):
                new_entry = copy(entry)
                new_entry.update({
                    text_key: var,
                    is_original_key: False
                })
                ostream.write(f"{json.dumps(new_entry)}\n")


def build_chain_translation_augmenter(language_chain: List[str], device: str) -> Sequential:
    pair_to_model = {
        "en-fr": "transformer.wmt14.en-fr",
        "en-de": "transformer.wmt19.en-de",
        "de-en": "transformer.wmt19.de-en",
        "en-ru": "transformer.wmt19.en-ru",
        "ru-en": "transformer.wmt19.ru-en"
    }
    if len(language_chain) <= 2:
        raise Exception("Can't backtranslate with less than two languages in a chain")

    augmenters = []
    for i in range(len(language_chain) - 2):
        from_key = f"{language_chain[i]}-{language_chain[i+1]}"
        to_key = f"{language_chain[i+1]}-{language_chain[i+2]}"
        from_model_name = pair_to_model[from_key]
        to_model_name = pair_to_model[to_key]
        augmenters.append(
            BackTranslationAug(from_model_name=from_model_name,
                               to_model_name=to_model_name,
                               device=device)
        )
    return Sequential(augmenters)


def read_original_data(path: Union[Path, str]):
    path = Path(path)
    with path.open('r') as istream:
        data = json.load(istream)
    result = []
    for k, v in data.items():
        v.update({'id': k, 'original': True})
        result.append(v)
    return result


if __name__ == "__main__":
    args = parse_args()

    data = read_original_data(args.input_json)
    
    augmenter = build_chain_translation_augmenter(args.chain, args.device)
    augment_dataset(augmenter, data, args.text_key, args.output_json, args.batch_size)
