from hw1 import Composer
import torch
from hw1 import Composer
from midi2seq import process_midi_seq, seq2piano
from torch.utils.data import DataLoader, TensorDataset
bsz = 100
epoch = 700

piano_seq = torch.from_numpy(process_midi_seq())
loader = DataLoader(TensorDataset(piano_seq), shuffle=True, batch_size=bsz, num_workers=4)

cps = Composer()
for i in range(epoch):
    for x in loader:
        cps.train(x[0].cuda(0).long())
        
midi = cps.compose(100)
midi = seq2piano(midi)
midi.write('piano1.midi')