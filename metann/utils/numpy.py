import numpy as np


def to_array(lst):  # force one-dimensional array
    array = np.empty(len(lst), dtype=object)
    for i, d in enumerate(lst):
        array[i] = d
    return array
