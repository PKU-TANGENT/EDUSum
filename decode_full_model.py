""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import argparse
import json
import os
from os.path import join
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op
from toolz.sandbox import unzip
from cytoolz import identity, concat, curry

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp
from rl import rl_edu_to_sentence, label, label_mask
from data.batcher import tokenize

from decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor
from decoding import make_html_safe

max_dec_edu = 10000
def decode(save_path, model_dir, split, batch_size,
           beam_size, diverse, max_len, cuda):
    start = time()
    # setup model
    with open(join(model_dir, 'meta.json')) as f:
        meta = json.loads(f.read())
    if meta['net_args']['abstractor'] is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        assert beam_size == 1
        abstractor = lambda x,y:x
    else:
        if beam_size == 1:
            abstractor = Abstractor(join(model_dir, 'abstractor'),
                                    max_len, cuda)
        else:
            print('BEAM')
            abstractor = BeamAbstractor(join(model_dir, 'abstractor'),
                                        max_len, cuda)
    extractor = RLExtractor(model_dir, cuda=cuda)

    # setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles
    dataset = DecodeDataset(split)

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )

    # prepare save paths and logs
    try:
        os.makedirs(join(save_path, 'output'))
    except:
        pass
    dec_log = {}
    dec_log['abstractor'] = meta['net_args']['abstractor']
    dec_log['extractor'] = meta['net_args']['extractor']
    dec_log['rl'] = True
    dec_log['split'] = split
    dec_log['beam'] = beam_size
    dec_log['diverse'] = diverse
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)

    # Decoding
    i = 0
    total_leng = 0
    total_num = 0
    with torch.no_grad():
        for i_debug, data_batch in enumerate(loader):
            raw_article_batch, sent_label_batch = tuple(map(list, unzip(data_batch)))
            tokenized_article_batch = map(tokenize(None), raw_article_batch)
            #ext_arts = []
            ext_inds = []
            dirty = []
            ext_sents = []
            masks = []
            for raw_art_sents, sent_labels in zip(tokenized_article_batch, sent_label_batch):
                ext = extractor(raw_art_sents, sent_labels)  # exclude EOE

                tmp_size = min(max_dec_edu, len(ext) - 1)
                #total_leng += sum([len(e) -1 for e in ext[:-1]])
                #total_num += len(ext) - 1
                #print(tmp_size, len(ext) - 1)
                ext_inds += [(len(ext_sents), tmp_size)]
                tmp_stop = ext[-1][-1].item()
                tmp_truncate = tmp_stop - 1
                str_arts = list(map(lambda x: ' '.join(x), raw_art_sents))
                for idx in ext[:tmp_size]:
                    t, m = rl_edu_to_sentence(str_arts, idx)
                    total_leng += len(t)
                    total_num += 1
                    assert len(t) == len(m)
                    if t == []:
                        assert len(idx) == 1
                        id = idx[0].item()
                        if id == tmp_truncate:
                            dirty.append(len(ext_sents))
                            ext_sents.append(label)
                            masks.append(label_mask)
                    else:
                        if idx[-1].item() != tmp_stop:
                            ext_sents.append(t)
                            masks.append(m)


                #ext_arts += [raw_art_sents[i] for i in ext]
            #print(ext_sents)
            #print(masks)
            #print(dirty)
            #exit(0)
            if beam_size > 1:
                #print(ext_sents)
                #print(masks)
                all_beams = abstractor(ext_sents, masks, beam_size, diverse)
                print('rerank')
                dec_outs = rerank_mp(all_beams, ext_inds)
                for d in dirty:
                    dec_outs[d] = []
                # TODO:!!!!!!!!!!!
            else:
                dec_outs = abstractor(ext_sents, masks)
                for d in dirty:
                    dec_outs[d] = []
            assert i == batch_size*i_debug
            for j, n in ext_inds:
                decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                with open(join(save_path, 'output/{}.dec'.format(i)),
                          'w') as f:
                    f.write(make_html_safe('\n'.join(decoded_sents)))
                if i % 100 == 0:
                    print(total_leng / total_num)
                i += 1

                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                    i, n_data, i/n_data*100,
                    timedelta(seconds=int(time()-start))
                ), end='')
    print()

_PRUNE = defaultdict(
    lambda: 2,
    {1:5, 2:5, 3:5, 4:5, 5:5, 6:4, 7:3, 8:3}
)

def rerank(all_beams, ext_inds):
    beam_lists = (all_beams[i: i+n] for i, n in ext_inds if n > 0)
    return list(concat(map(rerank_one, beam_lists)))

def rerank_mp(all_beams, ext_inds):
    beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    with mp.Pool(8) as pool:
        reranked = pool.map(rerank_one, beam_lists)
    return list(concat(reranked))

def rerank_one(beams):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)
    dec_outs = [h.sequence for h in best_hyps]
    return dec_outs

def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))

def _compute_score(hyps):
    all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
    lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    return (-repeat, lp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run decoding of the full model (RL)')
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--model_dir', help='root of the full model')

    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--val', action='store_true', help='use validation set')
    data.add_argument('--test', action='store_true', help='use test set')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--beam', type=int, action='store', default=1,
                        help='beam size for beam-search (reranking included)')
    parser.add_argument('--div', type=float, action='store', default=1.0,
                        help='diverse ratio for the diverse beam-search')
    parser.add_argument('--max_dec_word', type=int, action='store', default=30,
                        help='maximun words to be decoded for the abstractor')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    data_split = 'test' if args.test else 'val'
    decode(args.path, args.model_dir,
           data_split, args.batch, args.beam, args.div,
           args.max_dec_word, args.cuda)
