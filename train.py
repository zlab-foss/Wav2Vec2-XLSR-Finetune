import torch
from importlib import reload
import numpy as np
from tqdm.notebook import tqdm
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.data import DataModule
from src.wav2vec import Wav2vec


seed_everything(42)

## preprocessing
tokenizer = Wav2Vec2CTCTokenizer(
    "./fa-vocab.json", 
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    word_delimiter_token="|",
    do_lower_case=False
)

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, 
                                             sampling_rate=16000, 
                                             padding_value=0.0, 
                                             do_normalize=True, 
                                             return_attention_mask=True)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, 
                              tokenizer=tokenizer)



## data
data_dir = 'cv-fa-6.1/cv-corpus-6.1-2020-12-11/fa/'
# data_dir = 'shemo-fa/'
csv_dir = '/media/data/soroosh/dataset/ASR/' + data_dir

data_module = DataModule(processor, csv_dir, min_dur=1, max_dur=10, batch_size=16)
data_module.setup()


## model
model = Wav2vec(processor, max_epochs=10)


## trainer
logger = TensorBoardLogger(
    save_dir='logs/',
    name='cv',
)

checkpoint = ModelCheckpoint(dirpath='/media/data/soroosh/weights/wav2vec2-large-xlsr-persian-cv/', 
                             filename='{epoch}-{val_loss:.2f}', 
                             monitor='val_loss',
                             save_top_k=1, 
                             period=1)

lr_logger = LearningRateMonitor(logging_interval='step')
trainer = Trainer(benchmark=True, 
                  gpus=1, 
                  num_sanity_val_steps=0,
                  logger=logger, 
                  max_epochs=10,
                  callbacks=[checkpoint, lr_logger])


## train and save
trainer.fit(model, data_module)
trainer.save_checkpoint('weights/cv/last.ckpt')
