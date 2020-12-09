import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.utils.data as data
import sentencepiece as spm

from transformers import BertModel, BertConfig


class BertClassifier(nn.Module):
    def __init__(self, config, pad_id, num_classes):
        super(BertClassifier, self).__init__()
        
        bert_config = BertConfig(
            vocab_size=config.src_vocab_size,
            hidden_size=config.h_size,
            num_hidden_layers=config.n_layers,
            num_attention_heads=config.n_heads,
            intermediate_size=config.d_ff,
            hidden_dropout_prob = config.dropout,
            pad_token_id=pad_id,
        )
        self.tr = BertModel(config=bert_config)
        self.drop = nn.Dropout(config.dropout)
        self.out = nn.Linear(config.h_size, num_classes)
        
    def forward(self, x):
        attn = (x != 0).float()
        x = self.tr(
            input_ids=x,
            attention_mask=attn,
            return_dict=True
        )
        x = x.last_hidden_state.mean(dim=1)
        x = self.drop(x)
        x = self.out(x)
        return x


bos_id = 1
eos_id = 2
pad_id = 0


class BestUtilModel:
    def __init__(self, config, file_path, model_path, device):
        self.config = config
        self.device = device
        
        self.cmd_le = joblib.load(f'{file_path}/cmd_encoder')
        self.text_tokenizer = spm.SentencePieceProcessor(f'{file_path}/txt_bpe_clf.model')
        
        self.model = BertClassifier(self.config, pad_id, len(self.cmd_le.classes_))
        self.model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
        self.model.eval()
        self.model.to(device)
        
    def predict_many(self, texts, beam_width=5):
        text_enc = [self.text_tokenizer.encode(x) for x in texts]
        
        tokens = nn.utils.rnn.pad_sequence([torch.tensor([bos_id] + x[:self.config.max_src_len] + [eos_id]).long() for x in text_enc], 
                                           batch_first=True, padding_value=pad_id)
        
        pred_utils = []
        with torch.no_grad():
            tokens = tokens.to(self.device)
            logits = self.model(tokens).cpu().numpy()
            topk = np.argpartition(-logits, beam_width-1, axis=1)[:,:beam_width]

            for i in range(len(texts)):
                pred = list(zip(self.cmd_le.inverse_transform(topk[i]), logits[i, topk[i]]))
                pred_utils.append(pred) 
                
        return pred_utils