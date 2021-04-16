import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pytorch_lightning as pl


def display_time(sec):
    h = int(sec // 3600)
    m = int((sec - h*3600) // 60)
    s = int(sec - h*3600 - m*60)
    return f'{h} hours, {m} minutes, {s} seconds'
    

class CVData(Dataset):
    def __init__(self, manifest, processor, min_dur=None, max_dur=None):
        super().__init__()
        df = pd.read_csv(manifest)
        n1 = df.shape[0]
        d1 = sum(df['duration'])
        if min_dur is not None:
            df = df[df['duration'] >= min_dur]
        if max_dur is not None:
            df = df[df['duration'] <= max_dur]
        n2 = df.shape[0]
        d2 = sum(df['duration'])
        if n2 < n1:
            print(f'{n1-n2} samples | {display_time(d1-d2)} were filtered.')
        if n2 > 0:
            df = df.sort_values(by=['duration'])
            self.df = df.reset_index(drop=True)
        else:
            raise Exception('no samples')
            
        self.processor = processor
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        samp = self.df.loc[idx]
#         input_values = self.processor(librosa.load(samp['path'], sr=16000)[0], sampling_rate=16000).input_values[0]
        input_values = self.processor(np.load(samp['np_path']), sampling_rate=16000).input_values[0]
        with self.processor.as_target_processor():
            labels = self.processor(samp["sentence"]).input_ids
        return input_values, labels


class DataModule(pl.LightningDataModule):
    def __init__(self, processor, csv_dir, min_dur=None, max_dur=None, batch_size=4):
        super().__init__()
        self.processor = processor
        self.csv_dir = csv_dir
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.bs = batch_size
    
    def setup(self, stage=None):
        self.train = CVData(self.csv_dir + 'train.csv', self.processor, self.min_dur, self.max_dur)
        self.val = CVData(self.csv_dir + 'val.csv', self.processor)
        print('num train samples:',len(self.train), ' total duration:', display_time(sum(self.train.df['duration'])))
        print('num val samples:',len(self.val), ' total duration:', display_time(sum(self.train.df['duration'])))
        

    def collate(self, inputs):
        input_features = [{"input_values": sample[0]} for sample in inputs]
        batch = self.processor.pad(
            input_features,
            padding=True,
            max_length=None,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            label_features = [{"input_ids": sample[1]} for sample in inputs]
            labels_batch = self.processor.pad(
                label_features,
                padding=True,
                max_length=None,
                pad_to_multiple_of=None,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
        

    def train_dataloader(self):
        return DataLoader(self.train, 
                          batch_size=self.bs, 
                          shuffle=False, 
                          pin_memory=True, 
                          num_workers=self.bs, 
                          collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(self.val, 
                          batch_size=self.bs, 
                          shuffle=False, 
                          pin_memory=True, 
                          num_workers=self.bs, 
                          collate_fn=self.collate)