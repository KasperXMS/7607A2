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

# Hyper-parameters
learning_rate = 0.1
epochs = 12
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
validation_interval = 3
batch_size = 64

# dataset used
training_data_filepath = "./conll2003/train.txt"
validation_data_filepath = "./conll2003/valid.txt"
test_data_filepath = "./conll2003/test.txt"

training_data = dataset_build_with_batch(training_data_filepath, batch_size)
validation_data = dataset_build(validation_data_filepath)
testing_data = dataset_build(test_data_filepath)

# word_list to tensor conversion
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    if torch.cuda.is_available():
        return torch.tensor(idxs, dtype=torch.long).cuda()
    else:
        return torch.tensor(idxs, dtype=torch.long)


# validation or testing on the model checkpoint
def validate(epoch, data, model, word_to_ix, tag_to_ix, ix_to_tag):
    print()
    pred_right = 0
    total_num = 0
    f = open("result.txt", "a+")
    tag_ground_truth = []
    tag_prediction = []
    with torch.no_grad():
        print("Validating at epoch {}...".format(epoch))
        f.write("\nValidating at epoch {}...\n".format(epoch))
        valid_id = 1
        for sentence, tags in data:
            tag_ground_truth.append(tags)
            print("\rValidating sentence {}".format(valid_id), end='')
            total_num += 1
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)

            inputs = prepare_sequence(sentence, word_to_ix)
            tag_scores = model.forward(inputs)

            if torch.cuda.is_available():
                tag_scores = tag_scores.cpu().detach()

            pred_idx_tags = []
            for scores in tag_scores:
                idx_max = -1
                score_max = float("-inf")
                for i in range(len(scores)):
                    if scores[i] > score_max:
                        score_max = scores[i]
                        idx_max = i

                pred_idx_tags.append(idx_max)

            pred_tags = []
            for i in range(len(tags)):
                pred_tag = ix_to_tag[str(pred_idx_tags[i])]
                pred_tags.append(pred_tag)
                if tag_to_ix[tags[i]] == pred_idx_tags[i]:
                    pred_right += 1

                total_num += 1

            tag_prediction.append(pred_tags)
            valid_id += 1

        print()

    print(classification_report(tag_ground_truth, tag_prediction, mode='strict', scheme=IOB2))
    f.write(classification_report(tag_ground_truth, tag_prediction, mode='strict', scheme=IOB2))
    print("Total: {}, Right: {}, Accuracy: {:.2f}%".format(total_num, pred_right,
                                                           float(pred_right) / float(total_num) * 100.0))

    f.close()
    return float(pred_right) / float(total_num) * 100.0


def train():
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # load word & tag dictionaries
    word_to_ix = word_to_idx([training_data_filepath, validation_data_filepath, test_data_filepath])
    tag_to_ix, ix_to_tag = tag_to_idx(training_data_filepath)

    # model settings
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    if torch.cuda.is_available():
        model = model.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_length = len(training_data)
    y_list = []
    x_list = []

    # Training
    for epoch in range(epochs):
        print("Epoch {} training...".format(epoch))
        batch_id = 1
        for sentence, tags in training_data:
            print("\rTraining progress {:.2f}% -- batch {}/{}".format(float(batch_id) / float(train_length) * 100.0, batch_id, train_length), end='')

            # Clear the accumulated gradients in the torch
            model.zero_grad()

            # Forward
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
            tag_scores = model(sentence_in)

            # loss computation and backward propagation
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

            batch_id += 1

        # validation settings
        if (epoch + 1) % validation_interval == 0:
            acc = validate(epoch, validation_data, model, word_to_ix, tag_to_ix, ix_to_tag)
            y_list.append(acc)
            x_list.append((epoch + 1) / validation_interval)

        print()

    plt.plot(x_list, y_list)
    plt.savefig("acc_result.png")

    torch.save(model.state_dict(), "BasicLSTMTagger.pth")

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    train()