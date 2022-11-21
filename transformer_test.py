import torch
import torch.nn as nn
import torch.optim as optim
from model import TransformerModel
from model import generate_square_subsequent_mask
from utils import *
from seqeval.metrics import accuracy_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from seqeval.metrics import f1_score
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from matplotlib import pyplot as plt
import datetime
from train2 import validate
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMBEDDING_DIM = 128
HIDDEN_DIM = 128
nlayers = 4  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
validation_interval = 3
batch_size = 32

training_data_filepath = "./conll2003/train.txt"
validation_data_filepath = "./conll2003/valid.txt"
test_data_filepath = "./conll2003/test.txt"

training_data = dataset_build_with_batch(training_data_filepath, batch_size)
validation_data = dataset_build(validation_data_filepath)
testing_data = dataset_build(test_data_filepath)

word_to_ix = word_to_idx([training_data_filepath, validation_data_filepath, test_data_filepath])
tag_to_ix, ix_to_tag = tag_to_idx(training_data_filepath)

model = TransformerModel(len(word_to_ix), len(tag_to_ix), EMBEDDING_DIM, nhead, HIDDEN_DIM, nlayers, dropout).to(device)
model.load_state_dict(torch.load("BasicTransformerTagger.pth"))
print(model)

word_to_ix = word_to_idx([training_data_filepath, validation_data_filepath, test_data_filepath])
tag_to_ix, ix_to_tag = tag_to_idx(training_data_filepath)

with torch.no_grad():
    data = copy.deepcopy(testing_data)
    print()
    pred_right = 0
    total_num = 0
    tag_ground_truth = []
    tag_prediction = []
    total_loss = 0.0

    for i in range(len(data)):
        tag_ground_truth.append(data[i][1])
        data[i] = ([word_to_ix[t] for t in data[i][0]], [tag_to_ix[t] for t in data[i][1]])

    dataset_word = []
    dataset_tag = []
    for data_tuple in data:
        dataset_word.append(data_tuple[0])
        dataset_tag.append(data_tuple[1])

    with torch.no_grad():
        print("Testing...")
        for data_tuple in data:
            mask = generate_square_subsequent_mask(1).to(device)
            inputs = torch.tensor([data_tuple[0]], dtype=torch.long).to(device)
            y = torch.tensor([data_tuple[1]], dtype=torch.long).to(device)
            tag_scores = model(inputs, mask)
            loss_function = nn.CrossEntropyLoss()

            tag_scores_ = tag_scores.view(-1, tag_scores.shape[2])
            y_ = y.view(y.shape[0] * y.shape[1])

            loss = loss_function(tag_scores_, y_)
            total_loss += loss

            if torch.cuda.is_available():
                tag_scores = tag_scores.cpu().detach()
            pred_raw = torch.argmax(tag_scores, dim=2)
            pred = [ix_to_tag[str(int(t))] for t in pred_raw[0]]
            tag_prediction.append(pred)

    print(classification_report(tag_ground_truth, tag_prediction))
    print("Accuracy: ", accuracy_score(tag_ground_truth, tag_prediction))
    print("Precision: ", precision_score(tag_ground_truth, tag_prediction))
    print("Recall: ", recall_score(tag_ground_truth, tag_prediction))
    print("F1 score: ", f1_score(tag_ground_truth, tag_prediction))


f = open('output_transform.txt', 'w+')
for i in range(len(testing_data)):
    for j in range(len(testing_data[i][0])):
        f.write('{} {}\n'.format(testing_data[i][0][j], tag_prediction[i][j]))

f.close()
