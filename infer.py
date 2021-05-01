import torch
import librosa
import pandas as pd
import numpy as np
from transformers import Wav2Vec2Processor
import editdistance
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from nemo.collections.asr import modules
import logging
from ctcdecode import CTCBeamDecoder

from src.wav2vec import Wav2vec


def calc_cer(hyp, ref):
    score = editdistance.eval(list(hyp), list(ref))
    n_words = len(list(ref))
    return score / n_words

def calc_wer(hyp, ref):
    score = editdistance.eval(hyp.split(), ref.split())
    n_words = len(ref.split())
    return score / n_words

processor = Wav2Vec2Processor.from_pretrained('assets/processor/')
model = Wav2vec.load_from_checkpoint('weights/cv/last.ckpt', processor=processor)
model.eval()

vocab = list(processor.tokenizer.get_vocab().keys())
vocab[4] = ' '

gt = 'من غلام قمرم غیر قمر هیچ مگو پیش من جز سخن شمع و شکر هیچ مگو سخن رنج مگو جز سخن گنج مگو ور از این بی خبری رنج مبر هیچ مگو دوش دیوانه شدم عشق مرا دید و بگفت آمدم نعره مزن جامه مدر هیچ مگو  ' 
x = processor(librosa.load('assets/test.wav', sr=16000)[0], sampling_rate=16000).input_values
x = torch.tensor(x).float()
logits, _ = model(x)
preds = F.softmax(logits, dim=-1)

greedy = processor.tokenizer.decode(torch.argmax(preds, dim=-1)[0], group_tokens=False)
print('greedy cer:', calc_cer(greedy, gt), ' wer:', calc_wer(greedy, gt))
print(greedy, '\n')


def get_hyp(a, b):
    decoder = CTCBeamDecoder(
        vocab,
        model_path='assets/cv6.binary',
        alpha=a,
        beta=b,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=2,
        blank_id=0,
        log_probs_input=False
    )

    beam_results, beam_scores, timesteps, out_lens = decoder.decode(preds)
    res = []
    for beam, out_len in zip(beam_results, out_lens):
        res += [processor.tokenizer.decode(beam[0][:out_len[0]], group_tokens=False)]
    return res

alphas = np.linspace(0, 10, 101)
betas = np.linspace(0, 10, 101)
values = np.ones(shape=(10000, 4))
cnt = 0

for alpha in tqdm(alphas):
    for beta in betas:
        res = get_hyp(alpha, beta)
        values[cnt, 0] = alpha
        values[cnt, 1] = beta
        values[cnt, 2] = calc_cer(res, gt)
        values[cnt, 3] = calc_wer(res, gt)
        cnt += 1

df = pd.DataFrame(values, columns=['alpha', 'beta', 'cer'])
df.to_csv('beam_lm_alpha_beta_10.csv', index=False)
