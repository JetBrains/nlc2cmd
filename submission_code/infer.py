import sentencepiece as spm
import torch 
import numpy as np
from dataclasses import dataclass
from copy import copy

from preprocessing import clean_text
from best_ctx import BestCtxModel
from best_util import BestUtilModel

EXPERIMENT_NAME = '-'

import config_clf
util_config = config_clf.Config()

import config_ctx
ctx_config = config_ctx.Config()
    

bos_id = 1
eos_id = 2
pad_id = 0



class Predictor:
    def __init__(self, path):
        
        self.util_model = BestUtilModel(util_config, path, f'{path}/util_model.pth', 'cpu')
        self.ctx_model  = BestCtxModel(ctx_config,  path, f'{path}/ctx_model.pth', 'cpu')


    def predict_many(self, texts, result_cnt):
        
        alpha = 0.6
        n_utils = 5
        beam_width = 5
        
        text_cleaned = [clean_text(x) for x in texts]
        
        pred_utils = self.util_model.predict_many(text_cleaned, n_utils)
        
        result = []
        with torch.no_grad():
            for i in range(len(text_cleaned)):
                candidates = []
                for j in range(n_utils):
                    util, util_proba = pred_utils[i][j]
                    pred = self.ctx_model.predict(text_cleaned[i], util, beam_width)

                    for pred_cmd, ctx_proba in pred:
                        joined_proba = (1 - alpha) * util_proba + alpha * ctx_proba
                        candidates.append((pred_cmd,joined_proba))
                        
                
                candidates = sorted(candidates, key=lambda x: -x[1])[:result_cnt]
                candidates = [x[0] for x in candidates]
                result.append(candidates)
                
        return result


predictor = Predictor('/nlc2cmd/src/submission_code')
# predictor = Predictor('./')
