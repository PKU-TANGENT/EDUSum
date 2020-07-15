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

## Requirements
- **Python 3** (tested on python 3.6)
- [PyTorch](https://github.com/pytorch/pytorch) 1.10.0
    - with GPU and CUDA enabled installation (though the code is runnable on CPU, it would be way too slow)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [cytoolz](https://github.com/pytoolz/cytoolz)
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [pyrouge](https://github.com/bheinzerling/pyrouge) (for evaluation)

You can use the python package manager of your choice (*pip/conda*) to install the dependencies.
The code is tested on the *Linux* operating system.

## Data
We can not provide the segmented data due to the copyroght. 
If you want to train or evaluate our model on CNN/DailyMail dataset, 
you need to do the next several steps to get the dataset.
- Please follow the instructions [here](https://github.com/ChenRocks/cnn-dailymail) for downloading and preprocessing the CNN/DailyMail dataset. 
- Segment the articles into EDUs by:
```cd segment/src && python3 run.py --input_dir=[path/to/cnn&dailymail_finished_dir]```.
We use the code of [Toward Fast and Accurate Neural Discourse Segmentation](https://github.com/PKU-TANGENT/NeuralEDUSeg)
and modify it accoording to the structure of CNN/DailyMail dataset.
Note this step may cost several hours to several days because we need to segment nearly thirty thousands of articles.
- After segmenting the articles, there is a folder named segmented that contains three folders: train, val and test. 
Set the enviroment variable ```export DATA=path_to_segmented_dir```  
- Get the labels of dat by ```python3 make_extraction_labels.py``` 

Setup the official ROUGE packages at *[here](https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5)*.
Please specify the path to your ROUGE package by setting the environment variable
`export ROUGE=[path/to/rouge/directory]`.

## Training
After preparing the segmented and labeled data, yuo can train an EDUSum model yourself:
- pretrain the EDU fusion module and EDU Selection module seperately: 
```
python train_abstractor.py --path=[path/to/abstractor/model] --w2v=[path/to/word2vec/word2vec.128d.226k.bin]
python train_extractor_ml.py --path=[path/to/extractor/model] --w2v=[path/to/word2vec/word2vec.128d.226k.bin]
```
- Train the whole model end-to-end by RL:
```
python train_full_rl.py --path=[path/to/save/model] --abs_dir=[path/to/EDU_fusion_module] --ext_dir=[path/to/EDU_Selection_module]
```
## Evaluation
To evaluate the trained model, run:
```
python decode_full_model.py --path=[path/to/save/decoded/files] --model_dir=[path/to/pretrained] --beam=[beam_size] [--test/--val]
```
To compute ROUGE score, run:
```
python eval_full_model.py --[rouge/meteor] --decode_dir=[path/to/save/decoded/files]
```