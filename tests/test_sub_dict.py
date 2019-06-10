import pytest
import sys
sys.path.append('../')

from collections import OrderedDict
from metann.utils import SubDict

super_dict = OrderedDict(zip(range(0, 10), range(1, 11)))
sub_dict = SubDict(super_dict, keys=[2, 1, 0], keep_order=False)
sub_dict_ordered = SubDict(super_dict, keys=[2, 1, 0], keep_order=True)


def test_basic():
    super_dict = OrderedDict(zip(range(0, 10), range(1, 11)))
    sub_dict = SubDict(super_dict, keys=[2, 1, 0], keep_order=False)
    sub_dict_ordered = SubDict(super_dict, keys=[2, 1, 0], keep_order=True)
    print("sub dict: ", sub_dict)
    print("ordered sub dict: ", sub_dict_ordered)


def test_getitem():
    super_dict = OrderedDict(zip(range(0, 10), range(1, 11)))
    sub_dict = SubDict(super_dict, keys=[2, 1, 0], keep_order=False)
    assert sub_dict[2] == 3


def test_setitem():
    super_dict = OrderedDict(zip(range(0, 10), range(1, 11)))
    sub_dict = SubDict(super_dict, keys=[2, 1, 0], keep_order=False)
    sub_dict_ordered = SubDict(super_dict, keys=[2, 1, 0], keep_order=True)
    sub_dict_ordered[10] = 11
    assert super_dict[10] == 11
    sub_dict[2] = 4
    assert sub_dict_ordered[2] == 4


def test_delitem():
    super_dict = OrderedDict(zip(range(0, 10), range(1, 11)))
    sub_dict = SubDict(super_dict, keys=[2, 1, 0], keep_order=False)
    sub_dict_ordered = SubDict(super_dict, keys=[2, 1, 0], keep_order=True)
    del sub_dict[0]
    assert 0 not in super_dict.keys()
    assert 0 not in sub_dict_ordered.keys()


def test_iter():
    super_dict = OrderedDict(zip(range(0, 10), range(1, 11)))
    sub_dict = SubDict(super_dict, keys=[2, 1, 0], keep_order=False)
    sub_dict_ordered = SubDict(super_dict, keys=[2, 1, 0], keep_order=True)
    del sub_dict[0]
    assert set(iter(sub_dict)) == set([1, 2])
    assert list(iter(sub_dict_ordered)) == [1, 2]
    print("passed")


