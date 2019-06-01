import collections


class SubDict(collections.abc.MutableMapping):
    def __init__(self, super_dict: collections.abc.Mapping, keys=[], keep_order=True):
        self.super_dict = super_dict
        self.sub_keys = set(keys)
        self.update_keys()
        self.keep_order = keep_order

    def __getitem__(self, item):
        if item in self.sub_keys:
            return self.super_dict.__getitem__(item)
        else:
            raise KeyError(item)

    def __setitem__(self, key, value):
        self.sub_keys.add(key)
        return self.super_dict.__setitem__(key, value)

    def __delitem__(self, key):
        self.sub_keys.remove(key)
        try:
            del self.super_dict[key]
        except KeyError:
            pass

    def __contains__(self, key):
        self.update_keys()
        return self.sub_keys.__contains__(key)

    def __iter__(self):
        if self.keep_order:
            for i in self.super_dict:
                if i in self.sub_keys:
                    yield i
        else:
            for i in self.sub_keys:
                yield i

    def __len__(self):
        self.update_keys()
        return len(self.sub_keys)

    def __str__(self):
        return "SubDict("+str(list(self))+")"

    def update_keys(self):
        self.sub_keys = self.sub_keys.intersection(self.super_dict.keys())