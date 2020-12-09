import pandas as pd
import nlpaug.augmenter.word as naw
from tqdm import tqdm
from pathlib import Path
import json
import argparse

from parse_config import get_stopwords
from utils import find_free_file


def augment_dataframe(original, augmenter, input_key, batch_size=1):
	augmented = original.copy()
	augmented.rename({'origin': 'obtained_from'}, axis=1)
	if batch_size == 1:
		for (_, orig_row), (_, aug_row) in tqdm(zip(original.iterrows(), augmented.iterrows())):
			aug_row[input_key] = augmenter.augment(orig_row[input_key])
	else:
		for i in tqdm(range(0, len(original), batch_size)):
			batch = list(original[input_key].iloc[i:i+batch_size])
			for j, a in enumerate(augmenter.augment(batch)):
				augmented[input_key].iloc[i+j] = a
	augmented['origin'] = 'augmented'
	return augmented


def save_dataframe(dataframe, path):
	save_path = find_free_file(path)
	dataframe.to_csv(save_path)


def save_config(config, path):
	with find_free_file(path).open('w') as ostream:
		ostream.write(json.dumps(config))


def main(args):
	df = pd.read_csv(args.input_csv, index_col=0)
	kwargs = {
		'top_k': 10,
		'action': 'insert',
		'model_path': args.bert_path,
		'aug_min': 2,
		'aug_max': 4,
		'stopwords': get_stopwords(args.stopwords)
	}
	augmenter = naw.ContextualWordEmbsAug(device=args.device, **kwargs)
	augmented = augment_dataframe(df, augmenter, args.text_key, batch_size=2)
	augmented.to_csv(args.output_csv)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input-csv', type=Path)
	parser.add_argument('-o', '--output-csv', type=Path)
	parser.add_argument('-b', '--bert-path', type=str)
	parser.add_argument('-d', '--device', type=str)
	parser.add_argument('-s', '--stopwords', type=str)
	parser.add_argument('-t', '--text_key', type=str)
	args = parser.parse_args()
	main(args)
