from model_base import ComposerBase, CriticBase
import logging
from abc import abstractmethod
from midi2seq import process_midi_seq, seq2piano, random_piano, piano2seq, segment
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np


class Critic(CriticBase):
    def __init__(self, load_trained=False):
        '''
        :param load_trained
            If load_trained is True, load a trained model from a file.
            Should include code to download the file from Google drive if necessary.
            else, construct the model
        '''
        device = torch.device('cuda:0' if torch.cuda.is_available() else 
                              'mps:0' if torch.backends.mps.is_available() else 'cpu')
        logging.info('Using device:', device)

        if load_trained:
            logging.info('load model from file ...')

        def train(self, x):
            '''
            Train the model on one batch of data
            :param x: train data. For critic training, x will be a tuple of two tensors (data, label)
            :return: (mean) loss of the model on the batch
            '''
            pass

        def score(self, x):
            '''
            Compute the score of a music sequence
            :param x: a music sequence
            :return: the score between 0 and 1 that reflects the quality of the music: the closer to 1, the better
            '''
            pass


class Composer(ComposerBase):
    def __init__(self, load_trained=False):
        '''
        :param load_trained
            If load_trained is True, load a trained model from a file.
            Should include code to download the file from Google drive if necessary.
            else, construct the model
        '''
        if load_trained:
            logging.info('load model from file ...')

        def train(self, x):
            '''
            Train the model on one batch of data
            :param x: train data. For composer training, a single torch tensor will be given
            :return: (mean) loss of the model on the batch
            '''
            pass

        def compose(self, n):
            '''
            Generate a music sequence
            :param n: length of the sequence to be generated
            :return: the generated sequence
            '''
            pass