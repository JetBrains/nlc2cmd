import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import catalyst.dl as dl
import sentencepiece as spm
from sklearn.model_selection import train_test_split
import shutil
import os
from preprocessing import clean_text
from submission_code.tools import MtDataset, Transformer
import argparse


import sys
sys.path.append('../clai/utils')

SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

bos_id = 1
eos_id = 2
pad_id = 0

from submission_code import config_ctx
config = config_ctx.Config()


def train(dev_dir, logdir, device):
	if not config.joined_vocab:
		spm.SentencePieceTrainer.train(input=f'{dev_dir}/text',
		                               model_prefix=f'{dev_dir}/txt_bpe_ctx',
		                               model_type='bpe',
		                               vocab_size=config.src_vocab_size)
		spm.SentencePieceTrainer.train(input=f'{dev_dir}/cmd',
		                               model_prefix=f'{dev_dir}/cmd_bpe_ctx',
		                               model_type='bpe',
		                               vocab_size=config.tgt_vocab_size, )
		text_tokenizer = spm.SentencePieceProcessor(f'{dev_dir}/txt_bpe_ctx.model')
		cmd_tokenizer = spm.SentencePieceProcessor(f'{dev_dir}/cmd_bpe_ctx.model')

	else:
		spm.SentencePieceTrainer.train(input=f'{dev_dir}/all',
		                               model_prefix=f'{dev_dir}/all_bpe_ctx',
		                               model_type='bpe',
		                               vocab_size=config.src_vocab_size, )
		text_tokenizer = spm.SentencePieceProcessor(f'{dev_dir}/all_bpe_ctx.model')
		cmd_tokenizer = text_tokenizer

	train = pd.read_csv(f'{dev_dir}/train.csv', index_col=0)
	train = train.dropna()
	train['cmd_cleaned'] = train['cmd_cleaned'].apply(lambda cmd: cmd.replace('|', ' |'))
	train['util'] = train.cmd_cleaned.apply(lambda x: x.strip(' $()').split()[0])
	train = train[train.util != ']']
	train = train.reset_index(drop=True)

	mandf = pd.read_csv(f'{dev_dir}/man.csv', index_col=0)
	mandf['ctx']  = mandf.apply(make_ctx, axis=1)
	mandf = mandf.drop_duplicates(subset=('cmd'))
	mandf = mandf.set_index('cmd')

	train['ctx'] = train['util'].map(mandf.ctx)
	train.text_cleaned = train.text_cleaned + ' ' + train.ctx.fillna('')

	train['text_enc'] = train.text_cleaned.progress_apply(text_tokenizer.encode)
	train['cmd_enc'] = train.cmd_cleaned.progress_apply(cmd_tokenizer.encode)


	tdf = train[train.origin == 'original']
	tdf2 = train[train.origin != 'original']
	train, valid = train_test_split(tdf, test_size=500, random_state=SEED)
	train = pd.concat([train, tdf2]).reset_index(drop=True)

	train_ds = MtDataset(train.text_enc, train.cmd_enc, config, bos_id, eos_id, pad_id)
	valid_ds = MtDataset(valid.text_enc, valid.cmd_enc, config, bos_id, eos_id, pad_id)

	model = Transformer(config, pad_id)
	print('# params', sum(p.numel() for p in model.parameters() if p.requires_grad))

	loaders = {
		'train': data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
		'valid': data.DataLoader(valid_ds, batch_size=config.batch_size),
	}

	criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
	optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer_lr,
	                             weight_decay=config.weight_decay, amsgrad=True)
	callbacks=[
		dl.CheckpointCallback(config.num_epochs),
	]

	callbacks.append( dl.SchedulerCallback(mode="epoch") )
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.plateau_factor, patience=3, cooldown=2, threshold=1e-3, min_lr=1e-6)

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
		#     check=True
	)


def make_ctx(rec):
	options = eval(rec['options'])
	cmd = rec['cmd']
	synopsis = clean_text(rec['synopsis'])
	r = f'|{cmd} {synopsis}'
	for opt in options:
		short_flag = opt['short'][0] if len(opt['short']) > 0 else ''
		text = clean_text(opt['text'])
		r += f'|{short_flag} ' + ' '.join(text.split()[:5])
	return r


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("dev_dir", type=str)
	parser.add_argument("logdir", type=str)
	parser.add_argument('-d', '--device', type=str, default='cpu', required=False)
	args = parser.parse_args()
	train(args.dev_dir, args.logdir, args.device)