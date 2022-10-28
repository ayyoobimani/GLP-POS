Part-of-Speech tagger
==============

Code to train and test our POS tagger for different languages.


Installation:
--------

Tested with Python 3.6, Transformers 4.15.0, Torch 1.10.1.

Install Flair with its dependencies (see [their repo](https://github.com/flairNLP/flair)):

`pip install flair`

Download our modified version of Flair and substitute the orginal Flair folder with ours.

Data:
--------

GLP1: `GLP1/` folder

GLP2: `GLP2/` folder

GLP2 trained models: Given the size of the models, we will release them in our servers, in case of acceptance, to preserve anonimity.

Test: `TEST/` folder (Note: As described in the paper, we modified some test sets to accomodate the multiword tokens problem)

Train your model:
--------

Example:

`python3 train_pos_tagger.py --lang por --gpu 5 --train train_file.connlu --test test_file.conllu --epochs 30`

Prediction and evaluation:
--------

Example:

`python3 predict.py --gpu 0 --model model_path --test test_path`

