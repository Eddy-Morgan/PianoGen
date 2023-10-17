import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.info(f'Importing dependencies')

from model_base import ComposerBase, CriticBase
from midi2seq import dim
import torch
from torch.utils.data import Dataset 
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import gdown
import os

vocab_len = dim
device = torch.device('cuda:0' if torch.cuda.is_available() else 
                        'mps:0' if torch.backends.mps.is_available() else 'cpu')


class MidiCriticDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sequence, label =  self.data[idx][:-1] , self.data[idx][-1]
        return (sequence,label)
    

class CriticModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden=256, n_layers=3):
        super(CriticModel, self).__init__()

        self.num_stacked_layers = n_layers
        self.hidden_size = n_hidden

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)


        
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=n_hidden, 
                            num_layers=n_layers, 
                            batch_first=True, dropout=0.25)
        # Output layer
        self.linear = nn.Linear(n_hidden, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        
        embeddings = self.embedding_layer(x)
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        out = lstm_out[:, -1, :]
        out = self.linear(out)
        log_probs = F.sigmoid(out)
        return log_probs

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
        embedding_dim = 10
        n_hidden=64
        n_layers=3
        self.model = CriticModel(vocab_len,embedding_dim,n_hidden,n_layers)
        self.model.to(device)

        learning_rate = 0.0001
        self.loss_function = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if load_trained:
            output = 'critic.pth'
            if not os.path.isfile(output):
                logging.info('load model from file ...')
                url = 'https://drive.google.com/uc?id=1_VaJ3CqfgFBUIuCB4qX3z_Lj2wOVqnIb'
                gdown.download(url, output, quiet=False)

            state_dict = torch.load(output).state_dict()
            self.model.load_state_dict(state_dict)
            logging.info(self.model) 

    def train(self, batch, epoch_n):
        '''
        Train the model on one batch of data
        :param x: train data. For critic training, x will be a tuple of two tensors (data, label)
        :return: (mean) loss of the model on the batch
        '''
        self.model.train(True)
        b_loss = 0.0

        sequence_batch = batch[0].to(device, dtype=torch.long)
        label_batch = batch[1].to(device, dtype=torch.float32).view(-1, 1)

        log_probs = self.model(sequence_batch)

        loss = self.loss_function(log_probs, label_batch)
        b_loss += loss.item()
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logging.info(f'epoch {epoch_n} - loss for batch at {b_loss}')
        return b_loss

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
                X = torch.tensor(X)
            predicted = self.model(X.to(device).long())
        return predicted
    
    def accuracy(self, X_test, Y_test):
        sigmoid_probabilities = self.score(X_test)
        threshold = 0.5
        # Convert sigmoid probabilities to binary labels
        predicted = (sigmoid_probabilities >= threshold)
        arr = (predicted == Y_test)
        final_test_acc = sum(arr)/len(arr)
        return final_test_acc
    
    
class ComposerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden=256, n_layers=3):
        super(ComposerModel, self).__init__()

        self.num_stacked_layers = n_layers
        self.hidden_size = n_hidden
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=n_hidden, 
                            num_layers=n_layers, 
                            batch_first=True, dropout=0.25)
        self.linear1 = nn.Linear(n_hidden, 128)
        # Output layer
        self.linear2 = nn.Linear(128, vocab_size)


    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        
        embeddings = self.embedding_layer(x)
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        out = F.relu(self.linear1(lstm_out[:, -1, :]))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

class Composer(ComposerBase):
    def __init__(self, load_trained=False):
        '''
        :param load_trained
            If load_trained is True, load a trained model from a file.
            Should include code to download the file from Google drive if necessary.
            else, construct the model
        '''
        embedding_dim = 10
        n_hidden=256
        n_layers=3
        learning_rate = 0.0001
        self.loss_function = nn.NLLLoss() # for multi-class classification, you apply LogSoftmax to the output 
        self.model = ComposerModel(vocab_len,embedding_dim,n_hidden,n_layers)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if load_trained:
            output = 'composer_checkpoint.pth.tar'
            if not os.path.isfile(output):
                logging.info('load model from file ...')
                url = 'https://drive.google.com/uc?id=1RXS-N_utuCWiRlDkQuEgNZgrKkguaZvn'
                gdown.download(url, output, quiet=False)

            _ = self.load_checkpoint()
            # now individually transfer the optimizer parts...
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    def train(self, batch , epoch_n=None):
        '''
        Train the model on one batch of data
        :param x: train data. For composer training, a single torch tensor will be given
        :return: (mean) loss of the model on the batch
        '''
        self.model.train(True)

        # Step 1. Prepare the inputs to be passed to the model
        sequence_batch = batch[:,:-1].to(device).long()
        next_batch = batch[:,-1].to(device).long()

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        self.model.zero_grad()

         # Step 3. Run the forward pass, getting log probabilities over next
        # note
        log_probs = self.model(sequence_batch)
        # Step 4. Compute your loss function.
        loss = self.loss_function(log_probs, next_batch)

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        self.optimizer.step()

        logging.info(f'Epoch {epoch_n} - batch loss {loss.item()} ')
        return loss.item()
    
    def save_composer_model_checkpoint(self, num_epochs):
        '''
        Saves the trained model upon call
        '''
        state = {'epoch': num_epochs + 1, 'state_dict': self.model.state_dict(),
             'optimizer': self.optimizer.state_dict(), 'losslogger': None}
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

        self.model.train(False)
        self.model.eval()

        with torch.no_grad():
            prompt_sequence = torch.zeros((1, 51))
            prompt_sequence[:,50] = np.random.randint(0,dim)
            generated_sequence = prompt_sequence
            
            for _ in range(n):
                output = self.model(prompt_sequence.to(device).long())
                print(output)
                # New value to append
                new_value = torch.tensor([[[output]]], dtype=torch.float32)
                # Append the new value to the original tensor
                prompt_sequence = torch.cat((prompt_sequence, new_value), dim=1)
                prompt_sequence = prompt_sequence[:,1:,:]

                generated_sequence = torch.cat((generated_sequence, new_value), dim=1)
                
        generated_sequence = generated_sequence.reshape((-1,1)).flatten().astype(int)
        return generated_sequence