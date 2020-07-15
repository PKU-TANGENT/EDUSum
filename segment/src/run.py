#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 03/05/2018 2:56 PM
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
import random
import numpy as np
import tensorflow as tf
from config import parse_args
from api import prepare, train, evaluate, segment
import json


def segmentation(args, mode):
    logger.info('Start segmenting {} dataset'.format(mode))
    path = os.path.join(args.path, mode)
    article_path = os.path.join(path, 'article')
    result_path = os.path.join(path, 'result')
    try:
        os.mkdir(article_path)
    except:
        pass
    try:
        os.mkdir(result_path)
    except:
        pass
    data_list = os.listdir(f'{path}')
    for data_name in data_list:
        with open(os.path.join(path, data_name)) as f:
            data = json.load(f)
        with open(os.path.join(article_path, data_name), 'w') as f:
            for sent in data['article']:
                f.write(sent + '\n')
    args.input_files = os.path.join(article_path, os.listdir(article_path))
    args.result_dir = os.path.join(path, 'segment')
    segment(args)
    for data_name in data_list:
        try:
            os.remove(os.path.join(article_path, data_name))
            with open(os.path.join(path, data_name)) as f:
                data = json.load(f)
            with open(os.path.join(args.result_dir, data_name)) as f:
                article = f.readlines()
                data['article'] = article
            os.remove(os.path.join(args.result_dir, data_name))
            with open(os.path.join(result_path, data_name), 'w') as f:
                json.dump(data, f)
        except:
            pass
    os.rmdir(article_path)
    os.rmdir(args.result_dir)
    logger.info('Finish segmenting {} dataset'.format(mode))

if __name__ == '__main__':
    args = parse_args()

    logger = logging.getLogger("SegEDU")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    segmentation(args, 'train')
    segmentation(args, 'val')
    segmentation(args, 'test')

