"""
POS tagger 

$ python3 pos_tagger_xlmr.py --lang por --gpu 5 --train train_file.connlu --test test_file.conllu --epochs 30

"""
import os
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
parser = argparse.ArgumentParser()
parser.add_argument("--lang", default=None, type=str, required=True, help="Language (3 letters)")
parser.add_argument("--gpu", default=None, type=str, required=True, help="GPU number [0--7]")
parser.add_argument("--train", default=None, type=str, required=True, help="Train file")
parser.add_argument("--test", default=None, type=str, required=True, help="Test file")
parser.add_argument("--epochs", default=30, type=int, required=False, help="Number of epochs, default 30")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

from flair.embeddings import TransformerWordEmbeddings
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.optim import AdamW


def train(lang, train, test, epochs):

    # 1. get the corpus
    columns = {1: 'text', 3: 'upos'}

    data_folder = ''
    corpus: Corpus = ColumnCorpus(data_folder, 
                                columns,
                                train_file =   train,      
                                test_file  =   test,
                                dev_file   =   test,                           
                                comment_symbol="#",
                                column_delimiter="\t"
                                )

    # 2. what label do we want to predict?
    label_type = 'upos'

    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type)
    print(label_dict)

    embeddings = TransformerWordEmbeddings('xlm-roberta-base', pooling_operation='first_last', 
                                            allow_long_sentences=False, max_length=512, truncation=True) 
    # 4. initialize sequence tagger
    tagger = SequenceTagger(hidden_size=128,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=label_type,
                            use_crf=False)

    # 5. initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # 6. start training
    trainer.train(lang+'-upos-xlmr-final',
                learning_rate=0.0001,
                mini_batch_size=256,
                mini_batch_chunk_size=32,
                optimizer=AdamW,
                max_epochs=epochs,             
                patience=epochs, # so that it trains for a fixed amount of epochs, regardless the dev set
                checkpoint=True)


def main():
    lang = args.lang
    train(lang, args.train, args.test, args.epochs)
    print('Model saved in: '+lang+'-upos-xlmr-final')


if __name__ == "__main__":
    main()

