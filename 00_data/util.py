# 00_data/util.py
#
# Utility functions for data processing


# Imports
import numpy as np
import os
import torch
import torchaudio
from torchaudio.transforms import Resample
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import glob

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FS
from config import ASVSPOOF5_PROTOCOL_TRAIN
from config import ASVSPOOF5_PROTOCOL_DEV
from config import ASVSPOOF5_PROTOCOL_EVAL
from config import ASVSPOOF5_AUDIO_ROOT

# --- Utility functions ---
def parse_protocol(protocol_path):
    """
    Parse the protocol file and return a dictionary of file paths and labels
    """

    protocol = []
    keys = ["speaker_id", "flac_file_name", "gender", "codec", "codec_q", "codec_seed", "attack_tag", "attack_label", "key", "tmp"]

    with open(protocol_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        data = {}
        parts = line.strip().split()
 
        # Loop through keys and parts, and add to data
        assert len(parts) == len(keys), f"Expected {len(keys)} fields, got {len(parts)}: {line}"
        for key, value in zip(keys, parts):
            data[key] = value
        
        # Add data to protocol
        protocol.append(data)
    return protocol


def build_audio_index(audio_root):
    """
    Recursively scan audio_root for .flac files and build a lookup index
    mapping filename (no extension) -> full file path.
    """
    index = {}
    dupes = []
    for filepath in glob.iglob(os.path.join(audio_root, "**", "*.flac"), recursive=True):
        utt_id = os.path.splitext(os.path.basename(filepath))[0]
        if utt_id in index:
            dupes.append(utt_id)
        else:
            index[utt_id] = filepath
    if dupes:
        print(f"WARNING: {len(dupes)} duplicate utterance IDs found")
    return index
        
def load_audio(filepath, target_fs=FS):
    """
    Load an audio file, resample if necessary with torchaudio, return torch tensor
    """

    audio, fs = sf.read(filepath)
    audio = torch.from_numpy(audio).float()

    if fs != target_fs:
        resampler = Resample(fs, target_fs)
        audio = resampler(audio)

    return audio

def fix_length(audio, target_length, trim_method="random"):
    """
    Fix the length of an audio tensor to the target length

    If the audio is shorter than the target length, pad with zeros evenly on both sides

    If the audio is longer than the target length:
    ..and trim_method is "random", truncate the audio by randomly selecting a start index
    that allows for a full target length segment without exceeding the audio length

    ..and trim_method is "center", truncate the audio by indexing target_length from the center of the audio

    ..and trim_method is "start", truncate the audio by indexing target_length from the start of the audio
    """
    assert audio.ndim == 1, "Audio must be a 1D tensor"
    assert trim_method in ["random", "center", "start"], "Invalid trim method"
    assert target_length > 0, "Target length must be positive"

    if audio.shape[0] == target_length:
        return audio
    
    if audio.shape[0] < target_length:
        padding = target_length - audio.shape[0]
        padding_left = padding // 2
        padding_right = padding - padding_left
        audio = torch.nn.functional.pad(audio, (padding_left, padding_right))
    elif audio.shape[0] > target_length:
        if trim_method == "random":
            start_idx = torch.randint(0, audio.shape[0] - target_length + 1, (1,))            
            audio = audio[start_idx:start_idx + target_length]
        elif trim_method == "center":
            start_idx = (audio.shape[0] - target_length) // 2
            audio = audio[start_idx:start_idx + target_length]
        elif trim_method == "start":
            audio = audio[:target_length]
    return audio

class ASVSPOOF5Dataset(Dataset):
    """
    ASVspoof5 dataset
    """
    def __init__(self, protocol_path, audio_root, target_fs, target_length, trim_method="random"):
        self.protocol = parse_protocol(protocol_path)
        self.audio_index = build_audio_index(audio_root)
        self.target_fs = target_fs
        self.target_length = target_length
        self.trim_method = trim_method

    def __len__(self):
        return len(self.protocol)

    def __getitem__(self, idx):
        protocol_item = self.protocol[idx]
        audio_path = self.audio_index[protocol_item["flac_file_name"]]
        audio = load_audio(audio_path, target_fs=self.target_fs)
        audio = fix_length(audio, self.target_length, trim_method=self.trim_method)
        label = 0 if protocol_item["key"] == "bonafide" else 1
        label = torch.tensor(label, dtype=torch.long)
        return audio, label

# Running tests
if __name__ == "__main__":
    # pass
    # protocol = parse_protocol(ASVSPOOF5_PROTOCOL_TRAIN)
    # print(protocol[:5])

    # audio_index = build_audio_index("/Volumes/LPM03 storage/Datasets/Audio/asvspoof5/flac_D 3")
    # for i, (key, value) in enumerate(audio_index.items()):
    #     if i == 51:
    #         print(key, value)
    #         audio = load_audio(value, target_fs=4000)
    #         print(audio.shape)
    #         break

    # Test fix_length

    # # Case 1: exact length, should return unchanged
    # a = torch.arange(100).float()
    # out = fix_length(a, 100)
    # assert out.shape[0] == 100
    # assert torch.equal(out, a)
    # print("exact length: OK")

    # # Case 2: shorter than target, check padding is even (or off-by-one) and zeros are in the right place
    # a = torch.ones(10)
    # out = fix_length(a, 16)
    # assert out.shape[0] == 16
    # print("pad shorter -> 16:", out.tolist())
    # # expect 3 zeros, 10 ones, 3 zeros (padding=6, left=3, right=3)

    # # Case 3: odd padding amount, confirms left/right split behaves as expected
    # a = torch.ones(10)
    # out = fix_length(a, 15)
    # print("pad shorter -> 15 (odd pad):", out.tolist())
    # # padding=5, left=2, right=3 -- confirm this matches your intended convention

    # # Case 4: longer than target, trim_method="start"
    # a = torch.arange(20).float()
    # out = fix_length(a, 5, trim_method="start")
    # assert torch.equal(out, torch.arange(5).float())
    # print("trim start: OK ->", out.tolist())

    # # Case 5: longer than target, trim_method="center"
    # a = torch.arange(20).float()
    # out = fix_length(a, 5, trim_method="center")
    # print("trim center:", out.tolist())
    # # audio.shape[0]-target_length = 15, //2 = 7 -> expect [7,8,9,10,11]

    # # Case 6: longer than target, trim_method="random" -- run multiple times, confirm varying start + correct length
    # a = torch.arange(50).float()
    # starts = set()
    # for _ in range(20):
    #     out = fix_length(a, 10, trim_method="random")
    #     assert out.shape[0] == 10
    #     starts.add(out[0].item())
    # print("trim random: distinct start values observed:", sorted(starts))
    # # should see more than one distinct start value across 20 draws

    # # Case 7: edge case, audio length == target_length + 1 (smallest possible random range)
    # a = torch.arange(11).float()
    # out = fix_length(a, 10, trim_method="random")
    # assert out.shape[0] == 10
    # print("edge case len=target+1: OK ->", out.tolist())

    # Test ASVSPOOF5Dataset
    dataset = ASVSPOOF5Dataset(
        protocol_path=ASVSPOOF5_PROTOCOL_TRAIN,
        audio_root=os.path.join(ASVSPOOF5_AUDIO_ROOT, "flac_T"),
        target_fs=4000,
        target_length=int(FS * 4),  # 4s
        trim_method="random",
    )

    print(f"Dataset size: {len(dataset)}")

    audio, label = dataset[0]
    print(f"audio shape: {audio.shape}, dtype: {audio.dtype}")
    print(f"label: {label}, dtype: {label.dtype}")

    # Pull a handful more, check label distribution / shape consistency
    for idx in [1, 100, 1000, len(dataset) - 1]:
        audio, label = dataset[idx]
        assert audio.shape[0] == int(FS * 4), f"Unexpected length at idx {idx}: {audio.shape}"
        print(f"idx={idx}: shape={audio.shape}, label={label.item()}")

    # Check label distribution
    import random
    random.seed(0)
    sample_idxs = random.sample(range(len(dataset)), 500)
    labels = [dataset[i][1].item() for i in sample_idxs]
    print(f"bonafide: {labels.count(0)}, spoof: {labels.count(1)}")

    # Wrap in DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    audio_batch, label_batch = next(iter(loader))
    print(f"audio_batch shape: {audio_batch.shape}")
    print(f"label_batch: {label_batch}")