from nlpaug.augmenter.word import BackTranslationAug
from utils import deepflatten_sequence
from typing import List


class MultiVariantBackTranslationAug(BackTranslationAug):

    def __init__(self, from_model_name='transformer.wmt19.en-de', to_model_name='transformer.wmt19.de-en', 
                 from_model_checkpt='model1.pt', to_model_checkpt='model1.pt', tokenizer='moses', bpe='fastbpe', 
                 is_load_from_github=True, name='BackTranslationAug', device='cpu', force_reload=False, verbose=0,
                 n_predictions :int = 1, generation_kwargs = None):
        super().__init__(from_model_name='transformer.wmt19.en-de', to_model_name='transformer.wmt19.de-en', 
                 from_model_checkpt='model1.pt', to_model_checkpt='model1.pt', tokenizer='moses', bpe='fastbpe', 
                 is_load_from_github=True, name='BackTranslationAug', device='cpu', force_reload=False, verbose=0)
        self.n_predictions = n_predictions
        self.generation_kwargs = generation_kwargs or {}

    def substitute(self, text):
        translated = self.sample_n(
            self.model.from_model,
            text,
            beam=self.n_predictions,
            n_samples=self.n_predictions,
            **self.generation_kwargs
        )
        result = []
        for t in translated:
            nested = self.sample_n(
                self.model.to_model,
                t,
                beam=self.n_predictions,
                n_samples=self.n_predictions,
                **self.generation_kwargs
            ) # List[List[str]]
            backtranslated = list(set(
                deepflatten_sequence(nested)
            )) # List[str] with unique entries
            result.append(backtranslated)
        return result

    @staticmethod
    def sample_n(
        model, sentences: List[str], beam: int = 1, verbose: bool = False, n_samples: int = 1, **kwargs
    ) -> List[List[str]]:
        if isinstance(sentences, str):
            return MultiVariantBackTranslationAug.sample_n(
                model, [sentences], beam=beam, verbose=verbose, n_samples=n_samples, **kwargs
            )
        tokenized_sentences = [model.encode(sentence) for sentence in sentences]
        batched_hypos = model.generate(tokenized_sentences, beam, verbose, **kwargs)
        return [
            [model.decode(hypos[i]["tokens"]) for i in range(min(n_samples, len(hypos)))]
            for hypos in batched_hypos
        ]
