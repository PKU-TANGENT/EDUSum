"""produce the dataset with (psudo) extraction label"""
import os
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import multiprocessing as mp

from cytoolz import curry, compose
import numpy as np
from utils import count_data
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

def dfs(art_sents, abst, ext, ext_list, ban, f, depth):
    if depth == 3:
        return ext_list, f
    new_list = []
    new_pos = []
    for i, sent in enumerate(art_sents):
        #print(sent)
        if i not in ban:
            new_list.append(ext + sent)
            new_pos.append(i)
    if not new_list:
        return ext_list, f
    #print(new_list)
    rouges = np.asarray(list(map(compute_rouge_l(reference=abst, mode='r'),
                                 new_list)))

    position = np.argmax(rouges)
    if rouges[position] <= f:
        return ext_list, f
    select = new_pos[position]
    #print(select)
    ban.add(select)
    return dfs(art_sents, abst, ext + new_list[position], ext_list + [select], ban, rouges[position], depth + 1)


def my_get_extract_label(art_sents, abs_sents):
    extracted = []
    scores = []
    ban = set()
    for abst in abs_sents:
        select, score = dfs(art_sents, abst, [], [], ban, 0., 0)
        extracted.append(select)
        scores.append(score)
        #print(select)
        #print(score)

    tmp = set()
    for e in extracted:
        for x in e:
            assert x not in tmp
            tmp.add(x)
    #exit(0)
    return extracted, scores


'''
def my_get_extract_label(art_sents, abs_sents):
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
'''


def get_extract_label(art_sents, abs_sents):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    scores = []
    indices = list(range(len(art_sents)))

    for abst in abs_sents:
        rouges = np.asarray(list(map(compute_rouge_l(reference=abst, mode='r'),
                                     art_sents)))

        ext = max(indices, key=lambda i: rouges[i])
        indices.remove(ext)
        extracted.append(ext)
        scores.append(rouges[ext])
        if not indices:
            break
    extracted = list(map(lambda x: [x], extracted))
    return extracted, scores


@curry
def process(split, i):
    #try:
    data_dir = join(DATA_DIR, split)


    try:
        with open(join(data_dir, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        #if isinstance(ext[0], list):
        #    return
    except:
        return

    print(i)
    print(data_dir)
    tokenize = compose(list, _split_words)
    art_sents = tokenize(data['edu'])
    abs_sents = tokenize(data['abstract'])
    if art_sents and abs_sents: # some data contains empty article/abstract
        extracted, scores = my_get_extract_label(art_sents, abs_sents)
    else:
        extracted, scores = [], []
    data['extracted'] = extracted
    data['score'] = scores

    with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ':'))
    #except:
    #    print('!!!报错', i)
def label_mp(split):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(split))
    with mp.Pool() as pool:
        list(pool.imap_unordered(process(split),
                                 [i for i in range(290000)], chunksize=1000))
                                 #list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time()-start)))

def label(split):
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    n_data = count_data(data_dir)
    total_sum = total_len = 0
    for i in range(n_data + 1000):
        print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100*i/n_data),
              end='')
        with open(join(data_dir, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        tokenize = compose(list, _split_words)
        art_sents = tokenize(data['edu'])
        abs_sents = tokenize(data['abstract'])
        sent_label = data['sentence']
        extracted, scores = my_get_extract_label(art_sents, abs_sents)
        data['extracted'] = extracted
        data['score'] = scores
        total_sum += sum(scores)
        total_len += len(scores)
        if i == 100:
            print(total_sum / total_len)
            #exit(0)
        '''
        with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
            json.dump(data, f, indent=4)
        '''
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def main():
    #for split in ['val', 'train']:  # no need of extraction label when testing
    #    label_mp(split)
    label_mp('val')
    #label('train')
    #process('train', 850)
if __name__ == '__main__':
    main()
