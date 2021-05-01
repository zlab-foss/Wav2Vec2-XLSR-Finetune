import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import editdistance
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model, AdamW


def calc_wer(hyp, ref):
    score = editdistance.eval(hyp.split(), ref.split())
    n_words = len(ref.split())
    return score / n_words

def calc_cer(hyp, ref):
    score = editdistance.eval(list(hyp), list(ref))
    n_words = len(list(ref))
    return score / n_words


class Wav2vec(pl.LightningModule):
    def __init__(self, processor, lr=1e-4, min_lr=1e-8, max_epochs=100):
        super().__init__()
        self.lr = lr
        self.min_lr = min_lr
        self.max_epochs = max_epochs
        self.processor = processor
        
        self.model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53", 
            attention_dropout=0.1,
            hidden_dropout=0.1,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.1,
            gradient_checkpointing=True, 
            ctc_loss_reduction="mean", 
            ctc_zero_infinity=True,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer.get_vocab())
        )
        self.model.freeze_feature_extractor()
        
        
    def setup_beam_decoder(self, n_beams=10, lm_path=None, alpha=0.1, beta=0, n_jobs=5):
        from ctcdecode import CTCBeamDecoder
        
        vocab = list(self.processor.tokenizer.get_vocab().keys())
        vocab[4] = ' '
        
        self.beam_decoder = CTCBeamDecoder(
            vocab,
            model_path=lm_path,
            alpha=alpha,
            beta=beta,
            cutoff_prob=1.0,
            beam_width=n_beams,
            num_processes=n_jobs,
            blank_id=0,
            log_probs_input=False
        )

        
    def beam_decode(self, logits):
        preds = F.softmax(logits, dim=-1)
        beam_results, beam_scores, timesteps, out_lens = self.beam_decoder.decode(preds)
        res = []
        for beam, out_len in zip(beam_results, out_lens):
            res += [self.processor.tokenizer.decode(beam[0][:out_len[0]], group_tokens=False)]
        return res

    
    def forward(self, input_values, attention_mask=None, labels=None):
        pred = self.model(input_values=input_values, attention_mask=attention_mask, labels=labels)
        return pred.logits, pred.loss
    
    def compute_metrics(self, preds_str, labels):
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        labels_str = self.processor.batch_decode(labels, group_tokens=False)
        wer = np.mean([calc_wer(hyp, ref) for hyp,ref in zip(preds_str, labels_str)])
        cer = np.mean([calc_cer(hyp, ref) for hyp,ref in zip(preds_str, labels_str)])
        return wer, cer, labels_str
    
    def transcribe(self, file, mode='greedy'):
        x = self.processor(librosa.load(file, sr=16000)[0], sampling_rate=16000).input_values
        x = torch.tensor(x).to(self.device).float()
        logits, _ = self.forward(x)
        if mode == 'greedy':
            preds = torch.argmax(logits, dim=-1)
            return self.decode_with_metrics(preds)[0]
        return self.beam_decode(logits)[0]
        
    
    def decode_with_metrics(self, preds, labels=None):
        preds_str = self.processor.batch_decode(preds)
        if labels is not None:
            wer, cer, labels_str = self.compute_metrics(preds_str, labels)
            return preds_str, labels_str, wer, cer 
        return preds_str
    
    def step(self, batch, mode='train'):
        logits, loss = self.forward(**batch)
        preds = torch.argmax(logits, dim=-1)
        
        _, _, wer, cer = self.decode_with_metrics(preds, batch['labels'])
        self.log(mode+'_wer', wer)
        self.log(mode+'_cer', cer)
        self.log(mode+'_loss', loss.item())
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch)
    
    def validation_step(self, batch, batch_idx):
        self.step(batch, mode='val')
        
    def configure_optimizers(self):
#         opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        opt = AdamW(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs, eta_min=self.min_lr)
        return [opt], [sch]
    
    def count_parameters(self):
        return self.model.num_parameters()
