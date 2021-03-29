# import os
# pwd_path = os.path.abspath(os.path.dirname(__file__))
# wordset_path = os.path.join(pwd_path,"word_set.dic")
# stopword_path = os.path.join(pwd_path,"stopword.txt")

class Configer(object):
    def __init__(self):
        self.params = {}
        self.names = {}
        self.defaults = {}

    def __getitem__(self, key):
        if key not in self.params:
            assert "该参数不存在：{}".format(key)
        return self.params.get(key, self.defaults.get(key))

    def __setitem__(self, key, value):
        self.params.__setitem__(key, value)

    def define(self, key, value, name="", default=None):
        self.params.__setitem__(key, value)
        self.names.__setitem__(key, name)
        self.defaults.__setitem__(key, default)
