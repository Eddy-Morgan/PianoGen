import logging
from sklearn.model_selection import train_test_split
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

import torch
from hw1 import Composer, Critic, CriticModel, device, MidiCriticDataset
from midi2seq import process_midi_seq, random_piano, seq2piano
from torch.utils.data import DataLoader, TensorDataset, Dataset

bsz = 100
epoch = 700

## build data loader for critic model
piano_seq = torch.from_numpy(process_midi_seq())
good_labels = torch.ones((piano_seq.shape[0],1))
piano_data = torch.hstack((piano_seq,good_labels))

fake_midix = [random_piano() for i in range(20000)]
fake_seq = torch.from_numpy(process_midi_seq(all_midis=fake_midix))
bad_labels = torch.zeros((fake_seq.shape[0],1))
fake_data = torch.hstack((fake_seq,bad_labels))

critic_data = torch.vstack((piano_data,fake_data))

train_data, test_data = train_test_split(critic_data , test_size=0.2)

X_test = test_data[:,:-1]

Y_test = test_data[:,-1]
Y_test = Y_test.reshape((-1,1))

X_test = X_test.clone().detach().to(device).long()
Y_test = Y_test.clone().detach().to(device).long()
    
critic_loader = DataLoader(MidiCriticDataset(train_data), shuffle=True, batch_size=bsz)

ctc = Critic(load_trained=True)

# for i in range(epoch):
#     for x in critic_loader:
#         ctc.train(x,i)

# # ctc.save_model()

## check accuracy
final_test_acc = ctc.accuracy(X_test, Y_test)
print(final_test_acc)



# # build data loader for composer model
# piano_seq = torch.from_numpy(process_midi_seq())
# composer_train_loader = DataLoader(TensorDataset(piano_seq), shuffle=True, batch_size = bsz)

# cps = Composer(load_trained=False)

# ## training compose model
# for i in range(epoch):
#     for x in composer_train_loader:
#         cps.train(x[0].to(device).long(), i)

# cps.save_composer_model_checkpoint(epoch)

## compose a sequence of piano plays
# composed = cps.compose(25000)
# midi = seq2piano(composed)
# midi_file = "piano1.midi"
# midi.write(midi_file)

## scoring music sequences
# seq = process_midi_seq(all_midis=[midi_file])
# predictions = ctc.score(seq)
# logging.info(f'piano music scoring ::: {predictions}')