import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.info(f'Importing dependencies')

from model_base import ComposerBase, CriticBase
from midi2seq import process_midi_seq, random_piano
import torch
from torch.utils.data import Dataset 
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import gdown
import os
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda:0' if torch.cuda.is_available() else 
                        'mps:0' if torch.backends.mps.is_available() else 'cpu')


class CriticModel(nn.Module):
    def __init__(self, n_classes, n_input=1, n_hidden=256, n_layers=3):
        super().__init__()
        self.num_stacked_layers = n_layers
        self.hidden_size = n_hidden
        
        self.lstm = nn.LSTM(input_size=n_input, hidden_size=n_hidden, num_layers=n_layers, batch_first=True, dropout=0.7)
        # Output layer
        self.fc = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out

class Critic(CriticBase):
    def __init__(self, load_trained=False):
        logging.debug(f'Initializing model')
        '''
        :param load_trained
            If load_trained is True, load a trained model from a file.
            Should include code to download the file from Google drive if necessary.
            else, construct the model
        '''
        logging.info(f'device is {device.type}')
        self.model = CriticModel(2,1,64,3)
        self.model.to(device)

        learning_rate = 0.0001
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if load_trained:
            output = 'critic.pth'
            if not os.path.isfile(output):
                logging.info('load model from file ...')
                url = 'https://drive.google.com/uc?id=1Yla0ZkFQtPNZww8mdcPKDWNn7UfCDVJq'
                gdown.download(url, output, quiet=False)

            state_dict = torch.load(output).state_dict()
            self.model.load_state_dict(state_dict)
            logging.info(self.model) 

    def train(self, batch):
        '''
        Train the model on one batch of data
        :param x: train data. For critic training, x will be a tuple of two tensors (data, label)
        :return: (mean) loss of the model on the batch
        '''
        self.model.train(True)
        b_loss = 0.0

        sequence_batch , label_batch = batch[0] , batch[1]
        # convert to one_hot_encoded_label_batch for classification
        label_batch = torch.stack([self.one_hot_encode(label.item())
                                                    for label in label_batch])
        output = self.model(sequence_batch.to(device))
        loss = self.loss_function(output, label_batch.to(device))
        b_loss += loss.item()
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logging.info(f'loss for batch at {b_loss}')
        return b_loss
    
    def one_hot_encode(self, label):
        return torch.tensor([1, 0], dtype=torch.float) if label else torch.tensor([0, 1], dtype=torch.float)

    def save_model(self):
        '''
        Saves the trained model upon call
        '''
        torch.save(self.model, 'critic.pth')
        

    def score(self, X):
        '''
        Compute the score of a music sequence
        :param x: a music sequence
        :return: the score between 0 and 1 that reflects the quality of the music: the closer to 1, the better
        '''
        self.model.train(False)
        self.model.eval()
        with torch.no_grad():
            if not type(X) == torch.Tensor:
                X = X.reshape((-1,51,1))
                X = torch.tensor(X).float().to(device) 
            output = self.model(X.to(device))
            predicted_index = torch.argmax(output, dim=1)
            predicted_index ^= 1 # index 0 is good and index 1 is bad 
        return predicted_index


    
class MidiComposerDataset(Dataset):
    def __init__(self, piano_seq):
        self.x_sequence, self.y_next, self.labels = self.preprocess(piano_seq)

    def __len__(self):
        return len(self.y_next)
    
    def preprocess(self, piano_seq):
        piano_seq = piano_seq.numpy()
        labels  = np.unique(piano_seq)
        scaler = MinMaxScaler(feature_range=(0,1))

        # Fitting scaler with the complete space and transforming the whole dataset on the scaler
        normalized_sequence = scaler.fit_transform(piano_seq.reshape((-1,1))).reshape(piano_seq.shape)

        # normalized_labels = np.unique(normalized_sequence)

        X_train = normalized_sequence[:,:-1]
        X_train = X_train.reshape((-1,X_train.shape[1],1))

        Y_train = piano_seq[:,-1]
        Y_train = Y_train.reshape((-1,1))

        X_train = torch.tensor(X_train).float()
        Y_train = torch.tensor(Y_train).float()
    
        return X_train, Y_train , labels

    def one_hot_encode(self, note):
        return torch.tensor(note == self.labels).float()
        
    def __getitem__(self, idx):
        sequence = self.x_sequence[idx]
        action = self.y_next[idx][0].item()
        encode_action = self.one_hot_encode(action)
        return (sequence,encode_action)

    
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

class Composer(ComposerBase):
    def __init__(self, load_trained=False):
        '''
        :param load_trained
            If load_trained is True, load a trained model from a file.
            Should include code to download the file from Google drive if necessary.
            else, construct the model
        '''
        self.model = ComposerModel(302,1,256, 2)
        self.model.to(device)

        learning_rate = 0.0001
        self.loss_function = nn.CrossEntropyLoss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if load_trained:
            output = 'composer_checkpoint.pth.tar'
            if not os.path.isfile(output):
                logging.info('load model from file ...')
                url = 'https://drive.google.com/uc?id=1eFGzx51bGC_x8QStHiJaPw66I52jyIGW'
                gdown.download(url, output, quiet=False)

            _ = self.load_checkpoint()
            # now individually transfer the optimizer parts...
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    def train(self, batch):
        '''
        Train the model on one batch of data
        :param x: train data. For composer training, a single torch tensor will be given
        :return: (mean) loss of the model on the batch
        '''
        self.model.train(True)
        b_loss = 0.0
        sequence_batch , label_batch = batch['sequence'].to(device) , batch['action'].to(device)
        output = self.model(sequence_batch)
        loss = self.loss_function(output, label_batch)
        b_loss += loss.item()
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logging.info(f'loss for batch {b_loss}')
        return b_loss
    
    def save_composer_model_checkpoint(self, num_epochs, scaler, labels):
        '''
        Saves the trained model upon call
        '''
        state = {'epoch': num_epochs + 1, 'state_dict': self.model.state_dict(),
             'optimizer': self.optimizer.state_dict(), 'losslogger': None, 
             'scaler':scaler,
             'notes':labels}
        torch.save(state, "composer_checkpoint.pth.tar")

    def load_checkpoint(self, filename='composer_checkpoint.pth.tar', losslogger=None):
        # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
        start_epoch = 0
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            self.checkpoint = torch.load(filename)
            start_epoch = self.checkpoint['epoch']
            self.model.load_state_dict(self.checkpoint['state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            losslogger = self.checkpoint['losslogger']
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(filename, self.checkpoint['epoch']))
            self.model.to(device)
        else:
            print("=> no checkpoint found at '{}'".format(filename))
        return start_epoch, losslogger

    def compose(self, n):
        '''
        Generate a music sequence
        :param n: length of the sequence to be generated
        :return: the generated sequence
        '''
        labels = self.checkpoint['notes']
        scaler = self.checkpoint['scaler']

        self.model.train(False)
        self.model.eval()

        normalized_labels = scaler.transform(labels.reshape((-1,1))).reshape(labels.shape)
        with torch.no_grad():
            prompt_sequence = torch.zeros((50, 1))
            prompt_sequence[-1] = np.random.choice(normalized_labels)

            prompt_sequence = prompt_sequence.reshape((-1,prompt_sequence.shape[0],1))
            generated_sequence = prompt_sequence
            
            for _ in range(n):
                output = self.model(prompt_sequence.to(device))
                predicted_index = int(torch.argmax(output, dim=1))
                predicted_note = normalized_labels[predicted_index]
                # New value to append
                new_value = torch.tensor([[[predicted_note]]], dtype=torch.float32)
                # Append the new value to the original tensor
                prompt_sequence = torch.cat((prompt_sequence, new_value), dim=1)
                prompt_sequence = prompt_sequence[:,1:,:]

                generated_sequence = torch.cat((generated_sequence, new_value), dim=1)
                
        generated_sequence = np.rint(scaler.inverse_transform(generated_sequence.reshape((-1,1))))
        generated_sequence = generated_sequence.reshape((-1,1)).flatten().astype(int)
        return generated_sequence