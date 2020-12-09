from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import sacrebleu
from typing import Set


STEMMER = SnowballStemmer("english")


def jaccard_score(set_1: Set, set_2: Set):
    if len(set_1) == 0 and len(set_2) == 0:
        return 1.0
    intersection_size = len(set_1.intersection(set_2))
    union_size = len(set_1) + len(set_2) - intersection_size
    return intersection_size / union_size


def jaccard_ngrams(text_1: str, text_2: str, n: int = 1, stem: bool = False):
    
    def identity(words):
        return words

    def stem_words(words):
        return [STEMMER.stem(w) for w in words]

    stemming_fn = stem_words if stem else identity
    text_1_ngrams = set(
        ngrams(stemming_fn(word_tokenize(text_1)), n)
    )
    text_2_ngrams = set(
        ngrams(stemming_fn(word_tokenize(text_2)), n)
    )
    return jaccard_score(text_1_ngrams, text_2_ngrams)


def single_reference_sentence_bleu(reference: str, variant: str, stem: bool = False):

    def stem_sentence(sentence):
        return " ".join([
            STEMMER.stem(w)
            for w in word_tokenize(sentence)
        ])

    if stem:
        variant = stem_sentence(variant)
        reference = stem_sentence(reference)
    return sacrebleu.sentence_bleu(variant, [reference]).score
