""" batching """
import random
from collections import defaultdict

from toolz.sandbox import unzip
from cytoolz import curry, concat, compose
from cytoolz import curried
import os
import torch
import torch.multiprocessing as mp

# Batching functions
def coll_fn(data):
    source_lists, target_lists, mask_lists = unzip(data)
    # NOTE: independent filtering works because
    #       source and targets are matched properly by the Dataset
    sources = list(filter(bool, concat(source_lists)))
    targets = list(filter(bool, concat(target_lists)))
    masks = list(filter(bool, concat(mask_lists)))
    assert all(sources) and all(targets)
    return sources, targets, masks


def coll_fn_extract(data):
    def is_good_data(d):
        """ make sure data is not empty"""
        source_sents, extracts, sentences = d
        return source_sents and extracts and sentences
    batch = list(filter(is_good_data, data))
    assert all(map(is_good_data, batch))
    return batch

@curry
def tokenize(max_len, texts):
    return [t.lower().split()[:max_len] for t in texts]

def conver2id(unk, word2id, words_list):
    word2id = defaultdict(lambda: unk, word2id)
    return [[word2id[w] for w in words] for words in words_list]

@curry
def prepro_fn(max_src_len, max_tgt_len, batch):
    sources, targets, masks = batch
    sources = tokenize(max_src_len, sources)
    targets = tokenize(max_tgt_len, targets)
    masks = [m[:max_src_len] for m in masks]
    batch = list(zip(sources, targets, masks))
    return batch

@curry
def prepro_fn_extract(max_src_len, max_src_num, batch):
    def prepro(sample):
        source_sents, extracts, sentence = sample
        tokenized_sents = tokenize(max_src_len, source_sents)[:max_src_num]
        sentence = sentence[:max_src_num]
        cleaned_extracts = [list(filter(lambda e: e < len(tokenized_sents),
                                        ext)) for ext in extracts]
        cleaned_extracts = list(filter(lambda e: len(e) > 0, cleaned_extracts))
        return tokenized_sents, cleaned_extracts, sentence
    batch = list(map(prepro, batch))
    return batch

@curry
def convert_batch(unk, word2id, batch):
    sources, targets = unzip(batch)
    sources = conver2id(unk, word2id, sources)
    targets = conver2id(unk, word2id, targets)
    batch = list(zip(sources, targets))
    return batch

@curry
def convert_batch_copy(unk, word2id, batch):
    sources, targets, masks = map(list, unzip(batch))
    ext_word2id = dict(word2id)
    for source in sources:
        for word in source:
            if word not in ext_word2id:
                ext_word2id[word] = len(ext_word2id)
    src_exts = conver2id(unk, ext_word2id, sources)
    sources = conver2id(unk, word2id, sources)
    tar_ins = conver2id(unk, word2id, targets)
    targets = conver2id(unk, ext_word2id, targets)
    batch = list(zip(sources, src_exts, tar_ins, targets, masks))
    return batch

@curry
def convert_batch_extract_ptr(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts, sentences = sample
        id_sents = conver2id(unk, word2id, source_sents)
        return id_sents, extracts, sentences
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ff(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts = sample
        id_sents = conver2id(unk, word2id, source_sents)
        binary_extracts = [0] * len(source_sents)
        for ext in extracts:
            binary_extracts[ext] = 1
        return id_sents, binary_extracts
    batch = list(map(convert_one, batch))
    return batch


@curry
def pad_batch_tensorize(inputs, pad, cuda=True):
    """pad_batch_tensorize

    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    batch_size = len(inputs)
    max_len = max(6, max(len(ids) for ids in inputs))
    # TODO:magic number!!!!! 6是为了保证卷积的时候不会出错！！！
    tensor_shape = (batch_size, max_len)
    tensor = tensor_type(*tensor_shape)
    tensor.fill_(pad)
    for i, ids in enumerate(inputs):
        tensor[i, :len(ids)] = tensor_type(ids)
    return tensor


def multi_target_pad_batch_tensorize(inputs, pad, cuda=True):
    """pad_batch_tensorize

    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    #print(inputs)
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    batch_size = len(inputs)
    max_len = max(len(ids) for ids in inputs)
    tensor_shape = (batch_size, max_len, 2)
    # TODO:改掉！！
    tensor = tensor_type(*tensor_shape)
    tensor.fill_(pad)
    for i, ids in enumerate(inputs):
        if len(ids) > 0:
            tensor[i, :len(ids)] = tensor_type(ids)
    return tensor


@curry
def batchify_fn(pad, start, end, data, cuda=True):
    sources, targets = tuple(map(list, unzip(data)))

    src_lens = [len(src) for src in sources]
    tar_ins = [[start] + tgt for tgt in targets]
    targets = [tgt + [end] for tgt in targets]

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)

    fw_args = (source, src_lens, tar_in)
    loss_args = (target, )
    return fw_args, loss_args


@curry
def batchify_fn_copy(pad, start, end, data, cuda=True):
    sources, ext_srcs, tar_ins, targets, masks = tuple(map(list, unzip(data)))

    src_lens = [len(src) for src in sources]
    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]

    tar_ins = [[start] + tgt for tgt in tar_ins]
    targets = [tgt + [end] for tgt in targets]

    source = pad_batch_tensorize(sources, pad, cuda)
    mask = pad_batch_tensorize(masks, 0, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda)

    ext_vsize = ext_src.max().item() + 1
    fw_args = (source, src_lens, tar_in, ext_src, ext_vsize, mask)
    loss_args = (target, )
    return fw_args, loss_args


@curry
def batchify_fn_extract_ptr(pad, data, cuda=True):
    source_lists, targets, sentences = tuple(map(list, unzip(data)))
    src_nums = list(map(len, source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))
    sentences = pad_batch_tensorize(sentences, pad=-1, cuda=cuda)
    #print(sentences.size())
    #sentences = torch.tensor(sentences).to(sources[0].devide)
    #sentences = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), sentences))
    # PAD is -1 (dummy extraction index) for using sequence loss


    def unpack(input):
        leng, target = input
        new_target = []
        for t in target:
            new_target += t
            new_target.append(leng)
        new_target.append(leng + 1)
        return new_target
    targets = list(map(unpack, zip(list(map(lambda x: len(x), source_lists)), targets)))
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)

    #print(target)
    #exit(0)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda  # use 0 here for feeding first conv sentence repr.
    )
    fw_args = (sources, sentences, src_nums, tar_in)
    loss_args = (target, )
    return fw_args, loss_args

@curry
def batchify_fn_extract_ff(pad, data, cuda=True):
    source_lists, targets = tuple(map(list, unzip(data)))

    src_nums = list(map(len, source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))

    tensor_type = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    target = tensor_type(list(concat(targets)))

    fw_args = (sources, src_nums)
    loss_args = (target, )
    return fw_args, loss_args


def _batch2q(loader, prepro, q, single_run=True):
    epoch = 0
    while True:
        for batch in loader:
            q.put(prepro(batch))
        if single_run:
            break
        epoch += 1
        q.put(epoch)
    q.put(None)

class BucketedGenerater(object):
    def __init__(self, loader, prepro,
                 sort_key, batchify,
                 single_run=True, queue_size=8, fork=True):
        self._loader = loader
        self._prepro = prepro
        self._sort_key = sort_key
        self._batchify = batchify
        self._single_run = single_run
        if fork:
            ctx = mp.get_context('forkserver')
            self._queue = ctx.Queue(queue_size)
        else:
            # for easier debugging
            self._queue = None
        self._process = None

    def __call__(self, batch_size: int):
        def get_batches(hyper_batch):
            indexes = list(range(0, len(hyper_batch), batch_size))
            #if not self._single_run:
                # random shuffle for training batches
            #    random.shuffle(hyper_batch)
            #    random.shuffle(indexes)
            #hyper_batch.sort(key=self._sort_key)
            for i in indexes:
                batch = self._batchify(hyper_batch[i:i+batch_size])
                yield batch

        if self._queue is not None:
            ctx = mp.get_context('forkserver')
            self._process = ctx.Process(
                target=_batch2q,
                args=(self._loader, self._prepro,
                      self._queue, self._single_run)
            )
            self._process.start()
            while True:
                d = self._queue.get()
                if d is None:
                    break
                if isinstance(d, int):
                    print('\nepoch {} done'.format(d))
                    continue
                yield from get_batches(d)
            self._process.join()
        else:
            i = 0
            while True:
                for batch in self._loader:
                    yield from get_batches(self._prepro(batch))
                if self._single_run:
                    break
                i += 1
                print('\nepoch {} done'.format(i))

    def terminate(self):
        if self._process is not None:
            self._process.terminate()
            self._process.join()
