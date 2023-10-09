from model_base import ComposerBase, CriticBase
import logging
from abc import abstractmethod
from midi2seq import process_midi_seq, seq2piano, random_piano, piano2seq, segment
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset 
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import gdown

device = torch.device('cuda:0' if torch.cuda.is_available() else 
                        'mps:0' if torch.backends.mps.is_available() else 'cpu')

class CriticModel(nn.Module):
    def __init__(self, n_classes, n_input=1, n_hidden=256, n_layers=3):
        super().__init__()
        self.num_stacked_layers = n_layers
        self.hidden_size = n_hidden
        
        self.lstm = nn.LSTM(input_size=n_input, hidden_size=n_hidden, num_layers=n_layers, batch_first=True, dropout=0.7)
        # Output layer
        self.linear = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = lstm_out[:, -1, :]
        out = self.linear(out)
        return out
    
class CriticSourceData():
    @staticmethod
    def preprocess(seed):
        expert_seq = process_midi_seq(maxlen=50,n=15000, shuffle_seed=seed)
        fake_midix = [random_piano(seed) for i in range(20000)]
        fake_seq = process_midi_seq(all_midis=fake_midix,maxlen=50,n=15000)

        critic_data = np.zeros((expert_seq.shape[0] + fake_seq.shape[0], expert_seq.shape[1]+1))
        critic_data[:expert_seq.shape[0],:expert_seq.shape[1]] = expert_seq
        critic_data[expert_seq.shape[0]:,:expert_seq.shape[1]] = fake_seq
        critic_data[:expert_seq.shape[0],expert_seq.shape[1]] = 1

        train_sequences, test_sequences = train_test_split(critic_data , test_size=0.2)

        X_train = train_sequences[:,:51]
        X_train = X_train.reshape((-1,51,1))

        Y_train = train_sequences[:,51]
        Y_train = Y_train.reshape((-1,1))

        X_test = test_sequences[:,:51]
        X_test = X_test.reshape((-1,51,1))

        Y_test = test_sequences[:,51]
        Y_test = Y_test.reshape((-1,1))

        X_train = torch.tensor(X_train).float().to(device) 
        Y_train = torch.tensor(Y_train).float().to(device) 

        X_test = torch.tensor(X_test).float().to(device) 
        Y_test = torch.tensor(Y_test).float().to(device) 

        return X_train,Y_train,X_test,Y_test
    
class MidiCriticDataset(Dataset):
    def __init__(self, X_sequence, Y_critic):
        self.X_sequence = X_sequence
        self.Y_critic = Y_critic

    def __len__(self):
        return len(self.Y_critic)
        
    def __getitem__(self, idx):
        sequence, label =  self.X_sequence[idx] ,self.Y_critic[idx]
        label = torch.tensor([1, 0]).float() if label else torch.tensor([0, 1]).float()
        return dict(
            sequence = sequence,
            label = label
        )



class Critic(CriticBase):
    def __init__(self, load_trained=False):
        '''
        :param load_trained
            If load_trained is True, load a trained model from a file.
            Should include code to download the file from Google drive if necessary.
            else, construct the model
        '''    
        self.model = CriticModel(2,1,64,3)
        self.model.to(device)

        learning_rate = 0.0001
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if load_trained:
            logging.info('load model from file ...')
            url = 'https://drive.google.com/uc?id=1T0qNtWpvG-NO2lCOp_Y1YB6_Z3Xym0xn'
            output = 'critic.pth'
            gdown.download(url, output, quiet=False)

    def train(self, batch):
        '''
        Train the model on one batch of data
        :param x: train data. For critic training, x will be a tuple of two tensors (data, label)
        :return: (mean) loss of the model on the batch
        '''
        running_loss = 0.0
        sequence_batch , label_batch = batch['sequence'].to(device) , batch['label'].to(device)
        output = self.model(sequence_batch)
        loss = self.loss_function(output, label_batch)
        running_loss += loss.item()
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return running_loss


    def score(self, X):
        '''
        Compute the score of a music sequence
        :param x: a music sequence
        :return: the score between 0 and 1 that reflects the quality of the music: the closer to 1, the better
        '''
        self.model = torch.load('critic.pth')
        self.model.train(False)
        with torch.no_grad():
            output = self.model(X.to(device))
            predicted_index = torch.argmax(output, dim=1)
            predicted_index ^= 1 # index 0 is good and index 1 is bad 
        return predicted_index


    
class ComposerModel(nn.Module):
    def __init__(self, n_classes, n_input=1, n_hidden=256, n_layers=2):
        super().__init__()
        self.num_stacked_layers = n_layers
        self.hidden_size = n_hidden
        
        self.lstm = nn.LSTM(input_size=n_input, hidden_size=n_hidden, num_layers=n_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        # Output layer
        self.linear = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        # take only the last output
        out = lstm_out[:, -1, :]
        # produce output
        out = self.linear(self.dropout(out))
        return out
    


class MidiComposerDataset(Dataset):
    def __init__(self,labels, x_sequence, y_next):
        self.x_sequence = x_sequence
        self.y_next = y_next
        self.labels = labels

    def __len__(self):
        return len(self.y_next)

    def one_hot_encode(self, note):
        return torch.tensor(note == self.labels).float()
        
    def __getitem__(self, idx):
        action = self.y_next[idx][0].item()
        encode_action = self.one_hot_encode(action)
        return dict(
            sequence = self.x_sequence[idx],
            action = encode_action
        )


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