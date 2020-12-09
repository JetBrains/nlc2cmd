from pathlib import Path
from typing import Union
import nlpaug.augmenter.word as naw
from nlpaug.flow import Sequential, Sometimes
import yaml

from augmenters import MultiVariantBackTranslationAug
from filtering import FilterAugmented, ThresholdAcceptor
from metrics import single_reference_sentence_bleu


def read_config(config_path: Union[Path, str]):
    config_path = Path(config_path)
    with config_path.open('r') as istream:
        return yaml.load(istream, Loader=yaml.FullLoader)


def build_augmentation_pipeline(config_file: Path):
    config = read_config(config_file)
    flow = build_flow(config["flow"])
    return build_filtration(flow, config["filtration"])


def build_flow(flow_config):
    augmenters = []
    for key, value in flow_config.items():
        if key == "settings":
            continue
        augmenters.append(build_augmenter(key, value))
    settings = flow_config['settings']
    if not settings['is_random']:
        return Sequential(augmenters)
    return Sometimes(augmenters, aug_p=settings['aug_p'])


def build_augmenter(key, kwargs):
    AUGMENTER_CLASSES = {
        "MultiVariantBackTranslationAug": MultiVariantBackTranslationAug,
        "SynonymAug": naw.SynonymAug,
        "ContextualWordEmbsAug": naw.ContextualWordEmbsAug
    }
    if 'stopwords' in kwargs:
        kwargs['stopwords'] = get_stopwords(kwargs['stopwords'])
    return AUGMENTER_CLASSES[key](**kwargs)


def get_stopwords(stopwords):
    if isinstance(stopwords, list):
        return stopwords
    with open(stopwords, 'r') as istream:
        return [
            l.strip() for l in istream
            if l.strip() != ""
        ]


def build_filtration(flow, filtration_config):
    METRICS = {
        "bleu": single_reference_sentence_bleu
    }

    if filtration_config['metric_fn'] is None:
        return flow
    return FilterAugmented(
        flow,
        metric_fn=single_reference_sentence_bleu,
        metric_acceptor=ThresholdAcceptor(filtration_config['low'], filtration_config['high'])
    )
