"""produce the dataset with (psudo) extraction label"""
import os
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import multiprocessing as mp

from cytoolz import curry, compose
import numpy as np
#from utils import count_data
from metric import compute_rouge_l, compute_rouge_n
import numpy as np

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

glob_k = 2


def _split_words(texts):
    return map(lambda t: t.split(), texts)


from copy import deepcopy


def dfs(depth, l):
    if depth == 0:
        return [[]]
    tmp_l = []
    for i, x in enumerate(l):
        tmp = dfs(depth - 1, l[i + 1:])
        tmp1 = []
        for y in tmp:
            y_ = deepcopy(y)
            y_.append(x)
            tmp1.append(y_)
        tmp = tmp1
        tmp_l += tmp
    return tmp_l


def my_compose(l):
    comp = []
    for i in range(1, glob_k + 1):
        comp += dfs(i, list(range(len(l))))
    comp = list(map(lambda x: sorted(x), comp))
    tmp_l = []
    for x in comp:
        tmps = []
        for ind in x:
            tmps += deepcopy(l[ind])
        tmp_l.append(tmps)
    return tmp_l, comp


def get_extract_label(art_sents, abs_sents):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    scores = []
    new_art_sents, composed = my_compose(art_sents)
    indices = list(range(len(new_art_sents)))

    for abst in abs_sents:
        rouges = list(map(compute_rouge_n(reference=abst, mode='f'),
                          new_art_sents))
        ext = max(indices, key=lambda i: rouges[i])
        extracted.append(composed[ext])
        scores.append(rouges[ext])
    #print(extracted, scores)
    return extracted, scores

@curry
def process(split, i):
    print(i)
    #try:
    data_dir = join(DATA_DIR, split)
    print(data_dir)
    try:
        with open(join(data_dir, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        ext = data['extracted']
        if isinstance(ext[0], list):
            return
    except:
        return
    tokenize = compose(list, _split_words)
    art_sents = tokenize(data['edu'])
    abs_sents = tokenize(data['abstract'])
    if art_sents and abs_sents: # some data contains empty article/abstract
        extracted, scores = get_extract_label(art_sents, abs_sents)
    else:
        extracted, scores = [], []
    data['extracted'] = extracted
    data['score'] = scores

    with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ':'))


def label_mp(split):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(split))
    with mp.Pool() as pool:
        list(pool.imap_unordered(process(split),
                                 [i for i in range(290000)], chunksize=1000))
                                 #list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time()-start)))

def main():
    label_mp('train')
    label_mp('val')
if __name__ == '__main__':
    main()
