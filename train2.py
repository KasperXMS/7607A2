import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from model import LSTMTagger2
from utils import *
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from matplotlib import pyplot as plt
import datetime

# Hyper-parameters
learning_rate = 0.2
epochs = 50
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
validation_interval = 3
batch_size = 10

# global variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset used
training_data_filepath = "./conll2003/train.txt"
validation_data_filepath = "./conll2003/valid.txt"
test_data_filepath = "./conll2003/test.txt"

training_data = dataset_build_with_batch(training_data_filepath, batch_size ** 2)
validation_data = dataset_build_with_batch(validation_data_filepath, batch_size ** 2)
testing_data = dataset_build(test_data_filepath)


# word_list to tensor conversion
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    if torch.cuda.is_available():
        return torch.tensor(idxs, dtype=torch.long).to(device)


# validation or testing on the model checkpoint
def validate(epoch, data_, model, word_to_ix, tag_to_ix, ix_to_tag, report=False):
    data = copy.deepcopy(data_)
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
        print("Validating at epoch {}...".format(epoch + 1))
        for data_tuple in data:
            inputs = torch.tensor([data_tuple[0]], dtype=torch.long).to(device)
            y = torch.tensor([data_tuple[1]], dtype=torch.long).to(device)
            tag_scores = model(inputs)
            loss_function = nn.NLLLoss()

            tag_scores_ = tag_scores.view(-1, tag_scores.shape[2])
            y_ = y.view(y.shape[0] * y.shape[1])

            loss = loss_function(tag_scores_, y_)
            total_loss += loss

            if torch.cuda.is_available():
                tag_scores = tag_scores.cpu().detach()
            pred_raw = torch.argmax(tag_scores, dim=2)
            pred = [ix_to_tag[str(int(t))] for t in pred_raw[0]]
            tag_prediction.append(pred)

    # print((tag_ground_truth))
    # print((tag_prediction))
    print(classification_report(tag_ground_truth, tag_prediction, ))
    print("Validation loss: ", float(total_loss))

    return float(total_loss)


def train():
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # load word & tag dictionaries
    word_to_ix = word_to_idx([training_data_filepath, validation_data_filepath, test_data_filepath])
    tag_to_ix, ix_to_tag = tag_to_idx(training_data_filepath)

    for i in range(len(training_data)):
        training_data[i] = ([word_to_ix[t] for t in training_data[i][0]], [tag_to_ix[t] for t in training_data[i][1]])

    dataset_word = []
    dataset_tag = []
    for data_tuple in training_data:
        dataset_word.append(data_tuple[0])
        dataset_tag.append(data_tuple[1])

    torch_set = Data.TensorDataset(torch.tensor(dataset_word, dtype=torch.long), torch.tensor(dataset_tag, dtype=torch.long))
    loader = Data.DataLoader(
        dataset=torch_set,
        batch_size=batch_size,
        num_workers=2
    )

    # model settings
    model = LSTMTagger2(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), batch_size).to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    y_list = []
    x_list = []
    z_list = []
    min_valid_loss = float('inf')

    # Training
    for epoch in range(epochs):
        training_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(loader):
            model.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            tag_scores = model(batch_x)

            tag_scores = tag_scores.view(-1, tag_scores.shape[2])
            batch_y = batch_y.view(batch_y.shape[0] * batch_y.shape[1])
            loss = loss_function(tag_scores, batch_y)
            training_loss += loss
            print('\rEpoch {} batch {} / {} under training, loss = {}'.format(epoch + 1, i + 1, len(loader), training_loss), end='')
            loss.backward()
            optimizer.step()

        # validation settings
        valid_loss = validate(epoch, validation_data, model, word_to_ix, tag_to_ix, ix_to_tag, report=True)

        if (epoch + 1) % validation_interval == 0:
            x_list.append(epoch + 1)
            y_list.append(valid_loss.detach())
            z_list.append(training_loss.detach())

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            print("Saving model...")
            torch.save(model.state_dict(), "BasicLSTMTagger.pth")

        print()

    plt.plot(x_list, y_list, label='Validation loss')
    plt.plot(x_list, z_list, label='Training loss')
    plt.savefig("loss.png")

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    train()