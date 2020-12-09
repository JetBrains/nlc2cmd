import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.utils.data as data
import sentencepiece as spm

from tools import beam_search, Transformer


bos_id = 1
eos_id = 2
pad_id = 0


class BestCtxModel:
    def __init__(self, config, file_path, model_path, device):
        self.config = config
        self.device = device
        
        if not config.joined_vocab:
            self.text_tokenizer = spm.SentencePieceProcessor(f'{file_path}/txt_bpe_ctx.model')
            self.cmd_tokenizer = spm.SentencePieceProcessor(f'{file_path}/cmd_bpe_ctx.model')
        else:
            self.text_tokenizer = spm.SentencePieceProcessor(f'{file_path}/all_bpe_ctx.model')
            self.cmd_tokenizer = self.text_tokenizer
        
        self.model = Transformer(self.config, pad_id)
        self.model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
        self.model.eval()
        self.model.to(self.device)
        
        self.ctx = joblib.load(f'{file_path}/man_processed')
        
        
    def predict(self, text, util, beam_width):
        text = text + ' ' + self.ctx.get(util, '')
        text_enc = self.text_tokenizer.encode(text)
        tokens = torch.tensor([bos_id] + text_enc[:self.config.max_src_len] + [eos_id]).long()
        
        with torch.no_grad():
            pred = beam_search(tokens, self.model.tr, pad_id, bos_id, eos_id, max_len=self.config.max_tgt_len, k=beam_width)
        
        pred = [(self.cmd_tokenizer.decode(list(map(int, x))), proba) for x, proba in pred]
        return pred