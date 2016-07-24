import math, pickle, warnings, time, json, copy
import sympy as sy
import numpy as np
# from attrdict import AttrDict



def dump_resJSON(res, fname):
    with open(fname,'wb') as res_file:
        if type(res) == list:
            dumplist = [r.as_str() for r in res]
            json.dump(dumplist, res_file, indent=2, separators=(',', ': '))
            # json.dump([r.as_str() for r in res], res_file, indent=2, separators=(',', ': '))
        elif type(res) == Solution:
            # print "Solution type!"
            json.dump(res.as_str(), res_file, indent=2, separators=(',', ': '))
    # res_file.close()

def load_resJSON(fname):
    with open(fname,'r') as res_file:
        res = json.load(res_file)
        if type(res) == list:
            res_ = [Solution.from_strdict(r) for r in res]
        elif type(res) == dict:
            res_ = Solution.from_strdict(res)
        return res_


class Solution(dict):

    def __init__(self, *args):
        resdict = {}
        for dictionary in args: 
            for k, v in dictionary.items():
                resdict[k] = v 
        self.__dict__.update((str(k), v) for k, v in resdict.items())
        dict.__init__(self, resdict)

    def as_str(self):
        try:
            return {str(k): float(v) for k, v in self.items()}
        except:
            return {str(k): v for k, v in self.items()}

    # def as_var(self):
    #     return {sy.Symbol(k): v for k, v in self.items()}

    @classmethod
    def from_strdict(cls, strdict):
        self = cls({sy.Symbol(k): v for k, v in strdict.items()})
        return self

    def __str__(self):
        repr_ = '{'
        shift = -1
        for k, v in self.items():
            line = '{}: {}\n'.format(str(k), v)
            repr_ += (' '*shift + line)
            shift = abs(shift)
        # print "repr_:", repr_
        repr_ = repr_[:-1] + '}\n'
        return repr_

    def __repr__(self):
        return "\n" + self.__str__()
