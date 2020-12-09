import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.utils.data as data
import catalyst.dl as dl
import sentencepiece as spm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import shutil
import os
import argparse
from submission_code.tools import UtilDataset, BertClassifier


import sys
sys.path.append('../clai/utils')

SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

bos_id = 1
eos_id = 2
pad_id = 0

from submission_code import config_clf
config = config_clf.Config()


def train(dev_dir, logdir, device):
	train = pd.read_csv(f'{dev_dir}/train.csv', index_col=0)
	train['all_utils'] = train['cmd_cleaned'].apply(select_utils)
	train = train.loc[train.all_utils.apply(str.strip).apply(len) > 0]
	train['util'] = train['all_utils'].apply(lambda x: x.split()[0])
	train = train.dropna().reset_index(drop=True)

	spm.SentencePieceTrainer.train(input=f'{dev_dir}/text',
	                               model_prefix=f'{dev_dir}/txt_bpe_clf',
	                               model_type='bpe',
	                               vocab_size=config.src_vocab_size)
	text_tokenizer = spm.SentencePieceProcessor(f'{dev_dir}/txt_bpe_clf.model')

	cmd_le = LabelEncoder()

	train['text_enc'] = train.text_cleaned.progress_apply(text_tokenizer.encode)
	train['y'] = cmd_le.fit_transform(train['util'].values)

	tdf = train[train.origin == 'original']
	tdf2 = train[train.origin != 'original']
	train, valid = train_test_split(tdf, test_size=500, random_state=SEED)
	train = pd.concat([train, tdf2]).reset_index(drop=True)

	train_ds = UtilDataset(train.text_enc, train.y, config, bos_id, eos_id, pad_id)
	valid_ds = UtilDataset(valid.text_enc, valid.y, config, bos_id, eos_id, pad_id)

	model = BertClassifier(config, pad_id, len(cmd_le.classes_))
	print('# params', sum(p.numel() for p in model.parameters() if p.requires_grad))

	loaders = {
		'train': data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
		'valid': data.DataLoader(valid_ds, batch_size=config.batch_size),
	}

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer_lr,
	                             weight_decay=config.weight_decay, amsgrad=True)
	callbacks=[
		dl.CheckpointCallback(config.num_epochs),
		dl.AccuracyCallback(num_classes=len(cmd_le.classes_), topk_args=[1, 5])
	]

	if config.schedule == 'OneCycleLR':
		scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.optimizer_lr, epochs=config.num_epochs, steps_per_epoch=len(loaders['train']))
		callbacks.append( dl.SchedulerCallback(mode="batch") )

	elif config.schedule == 'ReduceLROnPlateau':
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.plateau_factor, patience=5, cooldown=3, threshold=1e-3, min_lr=1e-6)
		callbacks.append( dl.SchedulerCallback(mode="epoch") )

	shutil.rmtree(logdir, ignore_errors=True)
	os.makedirs(logdir, exist_ok=True)

	runner = dl.SupervisedRunner(device=device)
	runner.train(
		model=model,
		loaders=loaders,
		criterion=criterion,
		optimizer=optimizer,
		scheduler=scheduler if config.schedule else None,
		num_epochs=config.num_epochs,
		verbose=True,
		logdir=logdir,
		callbacks=callbacks,
	)
	joblib.dump(cmd_le, f'{dev_dir}/cmd_le')


def select_utils(text):
	text = text.replace('|', ' |')
	r = []
	for w in text.split():
		if w[0].isalpha() and w != 'ARG':
			w = w.strip(')')
			r.append(w)
	return ' '.join(r)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("dev_dir", type=str)
	parser.add_argument("logdir", type=str)
	parser.add_argument('-d', '--device', type=str, default='cpu', required=False)
	args = parser.parse_args()
	train(args.dev_dir, args.logdir, args.device)