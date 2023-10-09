from hw1 import Composer, CriticSourceData, MidiCriticDataset
from midi2seq import process_midi_seq, seq2piano
import torch
from torch.utils.data import DataLoader, TensorDataset

bsz = 100
epoch = 0

# build data loader for critic model
X_train,Y_train,X_test,Y_test = CriticSourceData.preprocess(1)
train_dataset = MidiCriticDataset(X_train,Y_train)
test_dataset = MidiCriticDataset(X_test,Y_test)

critic_train_loader = DataLoader(train_dataset,batch_size = bsz, shuffle=True)
critic_test_loader = DataLoader(test_dataset,batch_size = bsz, shuffle=False)

for i in range(epoch):
    for batch in critic_train_loader:
        pass

cps = Composer(load_trained=True)
# for i in range(epoch):
#     for x in loader:
#         cps.train(x[0].cuda(0).long())
        
midi = cps.compose(100)
midi = seq2piano(midi)
midi.write('piano1.midi')