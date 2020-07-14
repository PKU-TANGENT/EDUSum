import os
import json
import argparse


def segmentation(path):
    article_path = os.path.join(path, 'article')
    os.mkdir(article_path)
    data_list = os.listdir(f'{path}')
    for data_name in data_list:
        with open(data_name) as f:
            data = json.load(f)
        with open(os.path.join(article_path, data_name), 'w') as f:
            for sent in data['article']:
                f.write(sent + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='process data'
    )
    parser.add_argument('--extract_document', required=False, help='extract document from json file')
    parser.add_argument('--model', required=False, help='')
    parser.add_argument('--path', required=True, help='path to files')
    args = parser.parse_args()
    if args.extract_document:
        segmentation(os.path.join(args.path, 'train'))
        segmentation(os.path.join(args.path, 'val'))
        segmentation(os.path.join(args.path, 'test'))
    #elif args.