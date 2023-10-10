import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

from hw1 import Composer, Critic, MidiComposerDataset, MidiCriticDataset, CriticModel, device
from midi2seq import process_midi_seq, seq2piano
import torch
from torch.utils.data import DataLoader, TensorDataset

bsz = 100
epoch = 1

### build data loader for critic model
piano_seq = torch.from_numpy(process_midi_seq())
loader = DataLoader(TensorDataset(piano_seq), shuffle=True, batch_size=bsz)

# cps = Composer(load_trained=True)

# ctc = Critic(load_trained=True)

for batch in loader:
    print(batch[0].to(device).long().shape)
    # ctc.train(x)


### build data loader for composer model
# piano_seq = torch.from_numpy(process_midi_seq(shuffle_seed=3))
# composer_train_loader = DataLoader(MidiComposerDataset(piano_seq), shuffle=True, batch_size = bsz)

# cps = Composer(load_trained=True)

# ctc = Critic(load_trained=True)

### training critic model
# for i in range(epoch):
#     for batch in composer_train_loader:
#         cps.train(batch)

### training critic model
# for i in range(epoch):
#     for x in critic_train_loader:
#         ctc.train(x)
#         break

### compose a sequence of piano plays
# composed = cps.compose(25000)
# midi = seq2piano(composed)
# midi_file = "piano1.midi"
# midi.write(midi_file)

### scoring music sequences
# seq = process_midi_seq(all_midis=[midi_file])
# predictions = ctc.score(seq)
# logging.info(f'piano music scoring ::: {predictions}')