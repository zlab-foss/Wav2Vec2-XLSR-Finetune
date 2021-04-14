import os
import string
import six
import re
import glob
import hazm
from num2fawords import words, ordinal_words
from tqdm import tqdm
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torchaudio
import librosa
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2Processor


chars_to_ignore = [
    ",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�",
    "#", "!", "؟", "?", "«", "»", "،", "(", ")", "؛", "'ٔ", "٬",'ٔ', ",", "?", 
    ".", "!", "-", ";", ":",'"',"“", "%", "‘", "”", "�", "–", "…", "_", "”", '“', '„',
    'ā', 'š',
#     "ء",
]

# In case of farsi
chars_to_ignore = chars_to_ignore + list(string.ascii_lowercase + string.digits)

chars_to_mapping = {
    'ك': 'ک', 'دِ': 'د', 'بِ': 'ب', 'زِ': 'ز', 'ذِ': 'ذ', 'شِ': 'ش', 'سِ': 'س', 'ى': 'ی',
    'ي': 'ی', 'أ': 'ا', 'ؤ': 'و', "ے": "ی", "ۀ": "ه", "ﭘ": "پ", "ﮐ": "ک", "ﯽ": "ی",
    "ﺎ": "ا", "ﺑ": "ب", "ﺘ": "ت", "ﺧ": "خ", "ﺩ": "د", "ﺱ": "س", "ﻀ": "ض", "ﻌ": "ع",
    "ﻟ": "ل", "ﻡ": "م", "ﻢ": "م", "ﻪ": "ه", "ﻮ": "و", 'ﺍ': "ا", 'ة': "ه",
    'ﯾ': "ی", 'ﯿ': "ی", 'ﺒ': "ب", 'ﺖ': "ت", 'ﺪ': "د", 'ﺮ': "ر", 'ﺴ': "س", 'ﺷ': "ش",
    'ﺸ': "ش", 'ﻋ': "ع", 'ﻤ': "م", 'ﻥ': "ن", 'ﻧ': "ن", 'ﻭ': "و", 'ﺭ': "ر", "ﮔ": "گ",
        
    # "ها": "  ها", "ئ": "ی",
    "۱۴ام": "۱۴ ام",
        
    "a": " ای ", "b": " بی ", "c": " سی ", "d": " دی ", "e": " ایی ", "f": " اف ",
    "g": " جی ", "h": " اچ ", "i": " آی ", "j": " جی ", "k": " کی ", "l": " ال ",
    "m": " ام ", "n": " ان ", "o": " او ", "p": " پی ", "q": " کیو ", "r": " آر ",
    "s": " اس ", "t": " تی ", "u": " یو ", "v": " وی ", "w": " دبلیو ", "x": " اکس ",
    "y": " وای ", "z": " زد ",
    "\u200c": " ", "\u200d": " ", "\u200e": " ", "\u200f": " ", "\ufeff": " ",
}


def multiple_replace(text, chars_to_mapping):
    pattern = "|".join(map(re.escape, chars_to_mapping.keys()))
    return re.sub(pattern, lambda m: chars_to_mapping[m.group()], str(text))

def remove_special_characters(text, chars_to_ignore_regex):
    text = re.sub(chars_to_ignore_regex, '', text).lower() + " "
    return text



def normalize_batch(batch, chars_to_ignore=chars_to_ignore, chars_to_mapping=chars_to_mapping):
    chars_to_ignore_regex = f"""[{"".join(chars_to_ignore)}]"""
    text = batch["sentence"].lower().strip()

    _normalizer = hazm.Normalizer()
    text = _normalizer.normalize(text)
    text = multiple_replace(text, chars_to_mapping)
    text = remove_special_characters(text, chars_to_ignore_regex)
    text = re.sub(" +", " ", text)
    _text = []
    for word in text.split():
        try:
            word = int(word)
            _text.append(words(word))
        except:
            _text.append(word)
            
    text = " ".join(_text) + " "
    text = text.strip()

    if not len(text) > 0:
        return None
    
    batch["sentence"] = text
    
    return batch

def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def speech_file_to_array_fn(batch):
    target_sampling_rate = 16000
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, target_sampling_rate)
    
    
    batch["speech"] = speech_array
    batch["sampling_rate"] = target_sampling_rate
    batch["duration_in_seconds"] = len(batch["speech"]) / target_sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch


def prepare_dataset(batch, processor):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch




def get_datasets(train_csv_path, test_csv_path, processor, n_jobs=10, min_secs=3, max_secs=20, make_vocab=False):
    
    common_voice_train = load_dataset("csv", data_files={"train": train_csv_path}, delimiter="\t")["train"]
    common_voice_test = load_dataset("csv", data_files={"test": test_csv_path}, delimiter="\t")["test"]
    
    
    common_voice_train = common_voice_train.map(normalize_batch, 
                                                fn_kwargs={"chars_to_ignore": chars_to_ignore, 
                                                           "chars_to_mapping": chars_to_mapping})
    common_voice_test = common_voice_test.map(normalize_batch, 
                                              fn_kwargs={"chars_to_ignore": chars_to_ignore, 
                                                         "chars_to_mapping": chars_to_mapping})
    
    if make_vocab:
        vocab_train = common_voice_train.map(extract_all_chars, 
                                             batched=True, 
                                             batch_size=-1, 
                                             keep_in_memory=True, 
                                             remove_columns=common_voice_train.column_names)
        
        vocab_test = common_voice_train.map(extract_all_chars, 
                                            batched=True, 
                                            batch_size=-1, 
                                            keep_in_memory=True, 
                                            remove_columns=common_voice_test.column_names)

        special_vocab = ["<pad>", "<s>", "</s>", "<unk>", "|"]
        vocab_list = list(sorted(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0])))
        vocab_list = [vocab for vocab in vocab_list if vocab not in [" ", "\u0307"]]
        vocab_dict = {v: k for k, v in enumerate(special_vocab + vocab_list)}
        print(len(vocab_dict))
        print(vocab_dict)

        with open('fa-vocab.json', 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)
    
    
    common_voice_train = common_voice_train.map(speech_file_to_array_fn, 
                                                remove_columns=common_voice_train.column_names, 
                                                num_proc=n_jobs)
    common_voice_test = common_voice_test.map(speech_file_to_array_fn, 
                                              remove_columns=common_voice_test.column_names, 
                                              num_proc=n_jobs)

    
    print(f"Split sizes [BEFORE]: {len(common_voice_train)} train and {len(common_voice_test)} validation.")
    filter_by_duration = lambda batch: min_secs <= batch["duration_in_seconds"] <= max_secs
    common_voice_train = common_voice_train.filter(filter_by_duration, num_proc=n_jobs)
    common_voice_test = common_voice_test
    print(f"Split sizes [AFTER]: {len(common_voice_train)} train and {len(common_voice_test)} validation.")
    
    
    common_voice_train = common_voice_train.map(prepare_dataset, 
                                                fn_kwargs={'processor': processor},
                                                remove_columns=common_voice_train.column_names, 
                                                batch_size=8, num_proc=n_jobs, batched=True)
    common_voice_test = common_voice_test.map(prepare_dataset, 
                                              fn_kwargs={'processor': processor},
                                              remove_columns=common_voice_test.column_names, 
                                              batch_size=8, num_proc=n_jobs, batched=True)
    
    return common_voice_train, common_voice_test




@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    
    