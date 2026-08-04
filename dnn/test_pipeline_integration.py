# Test pipeline integration
#
# Dataset -> DataLoader -> Frontend

import sys, os
from torch.utils.data import DataLoader


from frontend import AudioFrontEnd
from data.util import ASVSPOOF5Dataset
from config import FS, ASVSPOOF5_PROTOCOL_TRAIN, ASVSPOOF5_AUDIO_ROOT

target_fs = 8000
target_bw = 2000

dataset = ASVSPOOF5Dataset(ASVSPOOF5_PROTOCOL_TRAIN, ASVSPOOF5_AUDIO_ROOT, target_fs=target_fs, target_length=int(target_fs*4))
loader = DataLoader(dataset, batch_size=4, shuffle=True)

audio_batch, label_batch = next(iter(loader))

afe = AudioFrontEnd(n_channels=8, bandwidth=2000, fs=target_fs)
out = afe(audio_batch)

print(f"input shape: {audio_batch.shape}")
print(f"output shape: {out.shape}, dtype: {out.dtype}")