import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from transformers import BertConfig, BertModel, EncoderDecoderConfig, EncoderDecoderModel


class MtDataset(data.Dataset):
    def __init__(self, src, tgt, config, bos_id, eos_id, pad_id):
        self.src = nn.utils.rnn.pad_sequence([
            torch.tensor([bos_id] + x[:config.max_src_len] + [eos_id]).long() for x in src], 
            batch_first=True, padding_value=pad_id)
        self.tgt = nn.utils.rnn.pad_sequence([
            torch.tensor([bos_id] + x[:config.max_tgt_len] + [eos_id]).long() for x in tgt], 
            batch_first=True, padding_value=pad_id)
        
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, i):
        return {
            'features': (self.src[i], self.tgt[i][:-1]), 
            'targets': self.tgt[i][1:]}


class Transformer(nn.Module):
    def __init__(self, config, pad_id):
        super(Transformer, self).__init__()
        
        encoder_config = BertConfig(
            vocab_size=config.src_vocab_size,
            hidden_size=config.h_size,
            num_hidden_layers=config.enc_layers,
            num_attention_heads=config.n_heads,
            intermediate_size=config.d_ff,
            hidden_dropout_prob = config.dropout,
            pad_token_id=pad_id,
        )
        decoder_config = BertConfig(
            vocab_size=config.tgt_vocab_size,
            hidden_size=config.h_size,
            num_hidden_layers=config.dec_layers,
            num_attention_heads=config.n_heads,
            intermediate_size=config.d_ff,
            hidden_dropout_prob = config.dropout,
            pad_token_id=pad_id,
            is_decoder=True,
            add_cross_attention=True,
        )
        encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        self.tr = EncoderDecoderModel(config=encoder_decoder_config)
        
        if config.joined_vocab:
            self.tr.encoder.embeddings.word_embeddings = self.tr.decoder.bert.embeddings.word_embeddings
        
    def forward(self, x):
        src, tgt = x
        src_attn = (src != 0).float()
        tgt_attn = (tgt != 0).float()
        x = self.tr(
            input_ids=src,
            attention_mask=src_attn,
            decoder_input_ids=tgt,
            decoder_attention_mask=tgt_attn,
        )
        x = x[0].permute(0,2,1)
        return x



def beam_search(src, model, pad_token, bos_id, end_token, max_len=10, k=5):
    device = next(model.parameters()).device
    src = src.view(1,-1).to(device)
    src_mask = (src != pad_token).to(device)
    
    memory = None
    
    input_seq = [bos_id]
    beam = [(input_seq, 0)] 
    for i in range(max_len):
        candidates = []
        candidates_proba = []
        for snt, snt_proba in beam:
            if snt[-1] == end_token:
                candidates.append(snt)
                candidates_proba.append(snt_proba)
            else:    
                snt_tensor = torch.tensor(snt).view(1, -1).long().to(device)
                
                if memory is None:
                    memory = model(
                        input_ids=src, 
                        attention_mask=src_mask,
                        decoder_input_ids=snt_tensor,
                    )
                else:
                    memory = model(
                        input_ids=src, 
                        attention_mask=src_mask,
                        decoder_input_ids=snt_tensor,
                        encoder_outputs=(memory[1], memory[-1]),
                    )
                    
                proba = memory[0].cpu()[0,-1, :]
                proba = torch.log_softmax(proba, dim=-1).numpy()
                best_k = np.argpartition(-proba, k - 1)[:k]

                for tok in best_k:
                    candidates.append(snt + [tok])
                    candidates_proba.append(snt_proba + proba[tok]) 
                    
        best_candidates = np.argpartition(-np.array(candidates_proba), k - 1)[:k]
        beam = [(candidates[j], candidates_proba[j]) for j in best_candidates]
        beam = sorted(beam, key=lambda x: -x[1])
        
    return beam


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


class UtilDataset(data.Dataset):
    def __init__(self, src, tgt, config, bos_id, eos_id, pad_id):
        self.src = nn.utils.rnn.pad_sequence([
            torch.tensor([bos_id] + x[:config.max_src_len] + [eos_id]).long() for x in src],
            batch_first=True, padding_value=pad_id)

        self.tgt = torch.tensor(tgt.values).long()

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return {'features': self.src[i], 'targets': self.tgt[i]}