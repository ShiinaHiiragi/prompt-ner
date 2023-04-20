import sys
import os
from utils.thulac import thulac

current_path = os.path.dirname(os.path.realpath(__file__))
assert os.path.exists(os.path.join(current_path, "./thulac/models"))
cutter = thulac(seg_only=True)

def cut(sentence):
    result = cutter.cut(sentence)
    return list(map(lambda item: item[0], result))
