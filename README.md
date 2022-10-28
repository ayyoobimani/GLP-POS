<<<<<<< HEAD
Part-of-Speech (POS) tagging is an important component of the NLP pipeline, but many low-resource languages lack labeled data for training. An established method for training a POS tagger in such a scenario is to create a labeled training set by transferring from high-resource languages. 
This repository contains the code, data, and trained models for the paper [Graph-Based Multilingual Label Propagation for Low-Resource Part-of-Speech Tagging](https://arxiv.org/abs/2210.09840).
The paper proposes a novel method for transferring labels from multiple high-resource source to low-resource target languages. We formalize POS tag projection as graph-based label propagation. Given translations of a sentence in multiple languages, we create a graph with words as nodes and alignment links as edges by aligning words for all language pairs. We then propagate node labels from source to target using a Graph Neural Network augmented with transformer layers. We show that our propagation creates training sets that allow us to train POS taggers for a diverse set of languages. When combined with enhanced contextualized embeddings, our method achieves a new state-of-the-art for unsupervised POS tagging of low resource languages.
=======
Graph-Based Multilingual Label Propagation for Low-Resource Part-of-Speech Tagging
==============
Part-of-Speech (POS) tagging is an important component of the NLP pipeline, but many low-resource languages lack labeled data for training. An established method for training a POS tagger in such a scenario is to create a labeled training set by transferring from high-resource languages. In this paper, we propose a novel method for transferring labels from multiple high-resource source to low-resource target languages. We formalize POS tag projection as graph-based label propagation. Given translations of a sentence in multiple languages, we create a graph with words as nodes and alignment links as edges by aligning words for all language pairs. We then propagate node labels from source to target using a Graph Neural Network augmented with transformer layers. We show that our propagation creates training sets that allow us to train POS taggers for a diverse set of languages. When combined with enhanced contextualized embeddings, our method achieves a new state-of-the-art for unsupervised POS tagging of low resource languages.
>>>>>>> 7a66aa472f506e8c5c59859c7d5e0709f26aacf2

Data:
--------

GLP1: `GLP1/` folder

GLP2: `GLP2/` folder

<<<<<<< HEAD
GLP2 trained models: Given the size of the models, we will release them in our servers [here](http://cistern.cis.lmu.de/glp_pos/) 
=======
GLP2 POS  models: Given the size of the models, you can find them [here](http://cistern.cis.lmu.de/glp_pos/) 
>>>>>>> 7a66aa472f506e8c5c59859c7d5e0709f26aacf2

Test: `TEST/` folder (Note: As described in the paper, we modified some test sets to accomodate the multiword tokens problem)

GLP models:
--------

Code info

Part-of-Speech taggers:
--------

Code to train and test our POS tagger for different languages.

Tested with Python 3.6, Transformers 4.15.0, Torch 1.10.1.

Install Flair with its dependencies (see [their repo](https://github.com/flairNLP/flair)):

`pip install flair`

Download our modified version of Flair and substitute the orginal Flair folder with ours.


Train POS model example:

`python3 train_pos_tagger.py --lang por --gpu 5 --train train_file.connlu --test test_file.conllu --epochs 30`

Prediction and evaluation of POS model example:

`python3 predict.py --gpu 0 --model model_path --test test_path`


Publication
--------

If you use the code, please cite 

```
<<<<<<< HEAD
TK
=======
@misc{https://doi.org/10.48550/arxiv.2210.09840,
  doi = {10.48550/ARXIV.2210.09840},  
  url = {https://arxiv.org/abs/2210.09840},
  author = {Imani, Ayyoob and Severini, Silvia and Sabet, Masoud Jalili and Yvon, François and Schütze, Hinrich},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Graph-Based Multilingual Label Propagation for Low-Resource Part-of-Speech Tagging},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

>>>>>>> 7a66aa472f506e8c5c59859c7d5e0709f26aacf2
``` 

License
-------

<<<<<<< HEAD
A full copy of the license can be found in LICENSE.
=======
A full copy of the license can be found in LICENSE.
>>>>>>> 7a66aa472f506e8c5c59859c7d5e0709f26aacf2
