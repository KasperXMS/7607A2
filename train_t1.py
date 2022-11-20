import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from model import TransformerModel
from model import generate_square_subsequent_mask
from utils import *
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from matplotlib import pyplot as plt
import datetime
from tqdm import tqdm

# Hyper-parameters
learning_rate = 0.2
epochs = 100
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
validation_interval = 3
batch_size = 32

# global variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset used
training_data_filepath = "./conll2003/train.txt"
validation_data_filepath = "./conll2003/valid.txt"
test_data_filepath = "./conll2003/test.txt"

training_data = dataset_build_with_batch(training_data_filepath, batch_size)
validation_data = dataset_build_with_batch(validation_data_filepath, batch_size)
testing_data = dataset_build(test_data_filepath)


# word_list to tensor conversion
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    if torch.cuda.is_available():
        return torch.tensor(idxs, dtype=torch.long).to(device)


# validation or testing on the model checkpoint
def validate(epoch, data_, model, word_to_ix, tag_to_ix, ix_to_tag, num_of_batches, report=False):
    data = copy.deepcopy(data_)
    print()
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

    # print((tag_ground_truth))
    # print((tag_prediction))
    print(classification_report(tag_ground_truth, tag_prediction, ))
    print("Validation loss: ", float(total_loss / num_of_batches))

    return float(total_loss / num_of_batches)


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
    model = TransformerModel(len(word_to_ix), len(tag_to_ix), EMBEDDING_DIM, nhead, HIDDEN_DIM, nlayers, dropout).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    y_list = []
    x_list = []
    z_list = []
    min_valid_loss = float('inf')

    # Training
    for epoch in range(epochs):
        training_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(tqdm(loader, desc='Training epoch {}'.format(epoch + 1))):
            mask = generate_square_subsequent_mask(len(batch_x)).to(device)

            model.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            tag_scores = model(batch_x, mask)

            tag_scores = tag_scores.view(-1, tag_scores.shape[2])
            batch_y = batch_y.view(batch_y.shape[0] * batch_y.shape[1])
            loss = loss_function(tag_scores, batch_y)
            training_loss += loss / len(loader)
            # print('\rEpoch {} batch {} / {} under training, loss = {}'.format(epoch + 1, i + 1, len(loader), training_loss), end='')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        # validation settings
        valid_loss = validate(epoch, validation_data, model, word_to_ix, tag_to_ix, ix_to_tag, len(loader), report=True)

        x_list.append(epoch + 1)
        y_list.append(valid_loss)
        if torch.cuda.is_available():
            training_loss = training_loss.cpu()
        z_list.append(training_loss.detach())

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            print("Saving model...")
            torch.save(model.state_dict(), "BasicTransformerTagger.pth")

        print()

    plt.plot(x_list, y_list, label='Validation loss')
    plt.plot(x_list, z_list, label='Training loss')
    plt.legend(loc="upper left")
    plt.grid(b=True, axis='y')
    plt.savefig("loss_transformer.png")

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    train()