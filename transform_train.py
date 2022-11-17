import torch
import math
from torch import nn, Tensor
from torch.utils.data import dataset
from typing import Tuple
import copy
import time
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from model import TransformerModel
from model import generate_square_subsequent_mask
from utils import *

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# dataset used
training_data_filepath = "./conll2003/train.txt"
validation_data_filepath = "./conll2003/valid.txt"
test_data_filepath = "./conll2003/test.txt"

training_data = dataset_build(training_data_filepath)
validation_data = dataset_build(validation_data_filepath)
testing_data = dataset_build(test_data_filepath)

word_to_ix = word_to_idx([training_data_filepath, validation_data_filepath, test_data_filepath])
tag_to_ix, ix_to_tag = tag_to_idx(training_data_filepath)


def data_process(in_data) -> Tuple[Tensor, Tensor]:
    """Converts raw text into a flat Tensor."""
    indexeds = []
    indexed = []
    i_targets = []
    i_target = []
    for item in in_data:
        for word in item[0]:
            indexed_word = word_to_ix[word]
            indexed.append(indexed_word)
        indexeds.append(indexed)
        indexed = []

        for tag in item[1]:
            indexed_target = tag_to_ix[tag]
            i_target.append(indexed_target)
        i_targets.append(i_target)
        i_target = []

    data = [torch.tensor(item, dtype=torch.long) for item in indexeds]
    target = [torch.tensor(item, dtype=torch.long) for item in i_targets]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data))), torch.cat(
        tuple(filter(lambda t: t.numel() > 0, target)))


# train_iter was "consumed" by the process of building the vocab,
# so we have to create it again
train_data, train_target = data_process(training_data)
val_data, val_target = data_process(validation_data)
test_data, test_target = data_process(testing_data)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def batchify(data: Tensor, target: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    target = target[:seq_len * bsz]
    target = target.view(bsz, seq_len).t().contiguous()
    return data.to(device), target.to(device)


batch_size = 20
eval_batch_size = 10
train_data, train_target = batchify(train_data, train_target, batch_size)  # shape [seq_len, batch_size]
val_data, val_target = batchify(val_data, val_target, eval_batch_size)
test_data, test_target = batchify(test_data, test_target, eval_batch_size)
bptt = 35


def get_batch(source_d: Tensor, source_f: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source_d) - 1 - i)
    data = source_d[i:i + seq_len]
    target = source_f[i:i + seq_len].reshape(-1)
    return data, target


ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, train_target, i)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model: nn.Module, eval_data: Tensor, eval_target: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, eval_target, i)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


best_val_loss = float('inf')
epochs = 3
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model, val_data, val_target)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)

    scheduler.step()
