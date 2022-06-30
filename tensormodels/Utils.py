import numpy as np
from tokenizer import Tokenizer
import multiprocessing as mp

global count
count=0

def multiprocess_func(data):
    global count
    tokenize = Tokenizer()
    paragraph_list = tokenize.auto_embedded(data)

    count+=1
    print("\r%d"%count,end="")

    return paragraph_list