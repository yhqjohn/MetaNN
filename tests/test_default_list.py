from metann.utils.containers import DefaultList


def test_default_list():
    l = DefaultList()
    l.fill([1, 2, 3, 4])
    for i, _ in zip(l, range(10)):
        print(i)
