import torch
import torch.nn as nn
import torch.optim as optim
from model import LSTMTagger
from utils import *
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from matplotlib import pyplot as plt
import datetime
from train import validate

training_data_filepath = "./conll2003/train.txt"
validation_data_filepath = "./conll2003/valid.txt"
test_data_filepath = "./conll2003/test.txt"

training_data = dataset_build(training_data_filepath)
validation_data = dataset_build(validation_data_filepath)
testing_data = dataset_build(test_data_filepath)

EMBEDDING_DIM = 64
HIDDEN_DIM = 64
word_to_ix = word_to_idx([training_data_filepath, validation_data_filepath, test_data_filepath])
tag_to_ix, ix_to_tag = tag_to_idx(training_data_filepath)

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
model.load_state_dict(torch.load("BasicLSTMTagger.pth"))

word_to_ix = word_to_idx([training_data_filepath, validation_data_filepath, test_data_filepath])
tag_to_ix, ix_to_tag = tag_to_idx(training_data_filepath)

with torch.no_grad():
    validate(0, testing_data, model, word_to_ix, tag_to_ix, ix_to_tag)