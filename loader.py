from collections import namedtuple
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
import sys
import torch
from inferno.extensions.metrics.categorical import CategoricalError
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.base import Callback
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

'''
trainset = np.load('/Users/WaxpEnterprises/Environments/DeepLearning/HW3/part1/template/dataset/wiki.train.npy')
validset = np.load('/Users/WaxpEnterprises/Environments/DeepLearning/HW3/part1/template/dataset/wiki.valid.npy')
vocabulary = np.load('/Users/WaxpEnterprises/Environments/DeepLearning/HW3/part1/template/dataset/vocab.npy')
print(trainset.shape[0])
print(validset.shape)
print(vocabulary.shape)
print(vocabulary[10])

batch_size = 3
'''


# Citation: https://raw.githubusercontent.com/cmudeeplearning11785/deep-learning-tutorials/master/recitation-6
# /shakespeare.py

'''
def iterator(data, args):
    batch_len = args.batch_len
    n = data.shape[0] // batch_len
    for i in range(n):
        # yield [data[i * batch_len: (i + 1) * batch_len, :-1], data[i * batch_len: (i + 1) * batch_len, 1:]]
        X = data[i * batch_len: (i + 1) * batch_len, :-1]
        Y = data[i * batch_len: (i + 1) * batch_len, 1:]
        yield (torch.from_numpy(X), torch.from_numpy(Y))
'''

def load_train():
    return np.load('./DeepLearningHW3P1/dataset/wiki.train.npy')


def load_valid():
    return np.load('./DeepLearningHW3P1/dataset/wiki.valid.npy')


def load_vocab():
    return np.load('./DeepLearningHW3P1/dataset/vocab.npy')


class Vocab_Loader(DataLoader):
    def __init__(self, dataset, shuffle, batch_size, batch_len):
        self.dataset = deepcopy(dataset)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.batch_len = batch_len

    def __iter__(self):
        self.array = deepcopy(self.dataset)
        if self.shuffle:
            self.array = np.random.shuffle(self.dataset)
        self.array = np.hstack(self.array)
        n = self.array.shape[0] // self.batch_size
        self.array = self.array[:n * self.batch_size]
        self.array = self.array.reshape((n, self.batch_size)).T
        m = self.array.shape[0] // self.batch_len
        for i in range(m):
            x = self.array[:-1, i * self.batch_len: (i + 1) * self.batch_len]
            y = self.array[1:, i * self.batch_len: (i + 1) * self.batch_len]
            print('X shape: ' + str(x.shape))
            print('Y shape:' + str(y.shape))
            yield [torch.from_numpy(x), torch.from_numpy(y)]


'''
def main():
    args = namedtuple('args',
                      [
                          'batch_size',
                          'save_directory',
                          'epochs',
                          'cuda',
                          'batch_len',
                          'embedding_dim',
                          'hidden_dim'])(
        10,
        'output/shakespeare',
        20,
        True,
        20,
        128,
        256)
    vocabulary = np.load('/Users/WaxpEnterprises/Environments/DeepLearning/HW3/part1/template/dataset/vocab.npy')
    validset = np.load('/Users/WaxpEnterprises/Environments/DeepLearning/HW3/part1/template/dataset/wiki.valid.npy')
    new_set = random_concatenation(validset)
    batches = batcher(new_set, args=args)
    pairs = iterator(batches, args=args)
    for each in pairs:
        print('X shape:' + str(each[0].size) + ', Y shape: ' + str(each[1].size))


if __name__ == '__main__':
    main()
'''
