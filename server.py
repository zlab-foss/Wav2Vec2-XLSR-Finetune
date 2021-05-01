import os
import numpy as np
from flask import Flask, jsonify, request
from scipy.io import wavfile
from transformers import Wav2Vec2Processor

from src.wav2vec import Wav2vec


processor = Wav2Vec2Processor.from_pretrained('assets/processor/')
model = Wav2vec.load_from_checkpoint('weights/cv/last.ckpt', processor=processor)
model.setup_beam_decoder(n_beams=100, lm_path='assets/cv6.binary', alpha=0.1, beta=0, n_jobs=5)
model.eval()

app = Flask(__name__)


@app.route('/speech')
def hello():
    return 'Hello World!'


@app.route('/speech/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        rec = file.read()
        rec = np.frombuffer(rec, dtype='float32')
        wavfile.write("req.wav", 16000, rec)
        greedy = model.transcribe('req.wav', mode='greedy')
        beam = model.transcribe('req.wav', mode='beam)
        os.remove("req.wav")
        return jsonify({'greedy': greedy, 'beam': beam})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8085)