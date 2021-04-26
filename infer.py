from transformers import Wav2Vec2Processor
from src.wav2vec import Wav2vec


processor = Wav2Vec2Processor.from_pretrained('processor/')
model = Wav2vec.load_from_checkpoint('weights/cv/last.ckpt', processor=processor)
model.cuda()
model.setup_beam_lm(n_beams=10, 
                    lm_path='cv6.arpa', 
                    alpha=0, 
                    beta=0, 
                    n_jobs=5)


print('greedy:')
print(model.transcribe('test.wav', mode='greedy'))
print('\n\nbeam seach with 3-gram:')
print(model.transcribe('test.wav', mode='beam'))