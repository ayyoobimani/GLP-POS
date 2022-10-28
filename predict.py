"""
Prediction:

python3 predict.py --gpu 0 --model model_path --test test_path

"""

import os
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, type=str, required=True, help="Model path")
parser.add_argument("--gpu", default=None, type=str, required=True, help="GPU number [0--7]")
parser.add_argument("--test", default=None, type=str, required=True, help="Test file")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

from flair.data import Corpus
from flair.datasets import ColumnCorpus, ColumnDataset
from flair.models import SequenceTagger

def predict(model_path, test_path):   

    columns = {1: 'text', 3: 'upos'}
    data_folder = ''
    corpus: Corpus = ColumnDataset(
                              column_name_map = columns,
                              path_to_column_file = test_path,
                              comment_symbol="#",
                              column_delimiter="\t"
                              )

    # Evaluation
    tagger: SequenceTagger = SequenceTagger.load(model_path+'/final-model.pt')
    result = tagger.evaluate(corpus, mini_batch_size=128, out_path=f"predictions.txt", gold_label_type='upos', num_workers=32)
    print(result.detailed_results) 

def main():
    model_path = args.model
    predict(model_path, args.test)
    print("Prediction saved in predictions.txt")

if __name__ == "__main__":
    main()