# EDUSum
This repository contains the code for our ACL 2020 paper:

*[Composing Elementary Discourse Units in Abstractive Summarization](https://www.aclweb.org/anthology/2020.acl-main.551/)*.


If you use this code, please cite our paper:
```
@inproceedings{li-etal-2020-composing,
    title = "Composing Elementary Discourse Units in Abstractive Summarization",
    author = "Li, Zhenwen  and
      Wu, Wenhao  and
      Li, Sujian",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020"
}
```
This project is based on [Chen's work](https://github.com/ChenRocks/fast_abs_rl).
## Requirements
- **Python 3** (tested on python 3.6)
- PyTorch 1.10.0
    - with GPU and CUDA enabled installation (though the code is runnable on CPU, it would be way too slow)
- Tensorflow 1.5.0
    - This is used for EDU segmentation.  
- gensim
- cytoolz
- spacy
- allennlp
- tensorboardX
- [pyrouge](https://github.com/bheinzerling/pyrouge) (for evaluation)


## Data
We can not provide the segmented data due to the copyright. 
If you want to train or evaluate our model on CNN/DailyMail dataset, 
you need to do the next several steps to get the dataset.
- Please follow the instructions [here](https://github.com/ChenRocks/cnn-dailymail) for downloading and preprocessing the CNN/DailyMail dataset. 
- Segment the articles into EDUs by:
```
cd segment/src && python3 run.py --input_dir=[path/to/cnn&dailymail_finished_dir]
```
We use the code of [Toward Fast and Accurate Neural Discourse Segmentation](https://github.com/PKU-TANGENT/NeuralEDUSeg)
and modify to meet the structure of CNN/DailyMail dataset.
Note this step may cost several hours to several days because we need to segment nearly 300,000 of articles.
- After segmenting the articles, there is a folder named *segmented* that contains three folders: *train, val and test*. 
Set the enviroment variable 
```
export DATA=[path/to/segmented_dir]
```  
- Get the labels of data by 
```
python3 make_extraction_labels.py
``` 

Setup the official ROUGE packages at *[here](https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5)*.
Please specify the path to your ROUGE package by setting the environment variable
```
export ROUGE=[path/to/rouge/directory]
```


## Training
After preparing the segmented and labeled data, yuo can train an EDUSum model yourself:
- pretrain the word2vec word embedding:
```
python train_word2vec.py --path=[path/to/word2vec]
```
- pretrain the EDU fusion module and EDU Selection module seperately: 
```
python3 train_abstractor.py --path=[path/to/abstractor/model] --w2v=[path/to/word2vec/word2vec.128d.226k.bin]
python3 train_extractor_ml.py --path=[path/to/extractor/model] --w2v=[path/to/word2vec/word2vec.128d.226k.bin]
```
- Train the whole model end-to-end by RL:
```
python3 train_full_rl.py --path=[path/to/save/model] --abs_dir=[path/to/EDU_fusion_module] --ext_dir=[path/to/EDU_Selection_module]
```
## Evaluation
To evaluate the trained model and generate summaries, run:
```
python3 decode_full_model.py --path=[path/to/save/decoded/files] --model_dir=[path/to/pretrained] --beam=[beam_size] [--test/--val]
```
To compute ROUGE score, firstly run:
```
python3 make_eval_references.py
```
to generate the reference files and then run
```
python3 eval_full_model.py --[rouge/meteor] --decode_dir=[path/to/save/decoded/files]
```