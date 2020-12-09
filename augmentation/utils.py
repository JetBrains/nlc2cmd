from functools import reduce
from typing import List, Any, Optional, Callable, Union
from pathlib import Path


def count_elements_in_nested_sequence(lst: List, sequence_type = List):
    if not isinstance(lst, sequence_type):
        return 1
    return sum(count_elements_in_nested_sequence(e, sequence_type) for e in lst)


def deepflatten_sequence(lst: List, sequence_type = List):
    """
    Flattens any nested sequence, for example:
    [[1, 2, 3], [[4], [5, [6], 7]]] -> [1, 2, 3, 4, 5, 6, 7]
    """
    def step(acc, elem):
        (lst, curr_idx) = acc
        if not isinstance(elem, sequence_type):
            lst[curr_idx] = elem
            return (lst, curr_idx + 1)
        return reduce(step, elem, acc)

    num_elements = count_elements_in_nested_sequence(lst, sequence_type)
    # to avoid reallocation
    result = [None] * num_elements
    result, _ = reduce(step, lst, (result, 0))
    return result


def maybe_apply(pred, arg, func):
    return func(arc) if pred(arg) else arg


def bind_func_as_method(
    instance: Any,
    func: Callable,
    method_name: Optional[str] = None
):
    if method_name is None:
        method_name = func.__name__
    bounded = func.__get__(instance, instance.__class__)
    setattr(instance, method_name, bounded)


def find_free_file(path: Union[str, Path]) -> Path:
    path = Path(path) if isinstance(path, str) else path
    new_path = path
    i = 2
    while new_path.exists():
        new_path = path.parent / f"{path.stem}_{i}{path.suffix}"
        i += 1
    return new_path
