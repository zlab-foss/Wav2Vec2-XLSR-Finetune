import numpy as np
import editdistance
from tqdm.notebook import tqdm

class Corrector:
    def __init__(self, vocab):
        self.vocab = sorted(vocab, key=lambda x: len(x))
        self.lens = {}
        for i, w in enumerate(self.vocab):
            n = len(w)
            if n not in self.lens:
                self.lens[n] = [i, i]
            else:
                self.lens[n][1] = i
        
    def get_candidates(self, min_len=1, max_len=20):
        inds = []
        for i in range(min_len, max_len+1):
            if i in self.lens:
                inds += range(self.lens[i][0], self.lens[i][1] + 1)
        res = [self.vocab[i] for i in inds]
        return res
        
    def replace_word(self, word):
        if word in self.vocab:
            return word
        
        n = len(word)
        candidates = self.get_candidates(n, n)
        dists = [editdistance.eval(list(cand), list(word))/ n for cand in candidates]
        return candidates[np.argmin(dists)]
    
    def edit(self, sent):
        res = []
        for w in tqdm(sent.split()):
            res += [self.replace_word(w)]
        return ' '.join(res)

            