from nlpaug import Augmenter
from typing import Callable, TypeVar


T = TypeVar("T")


class FilterAugmented:

    def __init__(
        self,
        augmenter: Augmenter,
        metric_fn: Callable[[str, str], T],
        metric_acceptor: Callable[[T], bool]
    ):
        self.augmenter = augmenter
        self.metric_fn = metric_fn
        self.metric_acceptor = metric_acceptor

    def augment(self, text: str, *args, **kwargs):
        augmented = self.augmenter.augment(text, *args, **kwargs)
        if isinstance(augmented, str):
            augmented = [augmented]
        filter_fn = lambda variant: self.metric_acceptor(
            self.metric_fn(text, variant)
        )
        return list(filter(filter_fn, augmented))


class ThresholdAcceptor:

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, value):
        return self.low < value < self.high
