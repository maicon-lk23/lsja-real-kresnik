# -*- coding:utf-8 -*-

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import librosa
import os
import numpy as np
import torch
import pandas as pd
from utils import combine_jamo, remove_lonely_jamo


processor = Wav2Vec2Processor.from_pretrained("42MARU/ko-spelling-wav2vec2-conformer-del-1s")
model = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean").to("cuda")
model.lm_head = torch.nn.Linear(1024, len(tokenizer.vocab))
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("42MARU/ko-spelling-wav2vec2-conformer-del-1s")

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"


model.load_state_dict(torch.load("./model.pt"))
model = model.to(device)

TESTPATH = "C:\\Users\\Admin\\Desktop\\work\\kohrepo\\kdh-wav2vec\\test_denoise"

pred_list = []
model.eval()
for i in tqdm(range(6000)):
    array, sampling_rate = librosa.load(os.path.join(TESTPATH, f"denoise_{i}.wav"))
    array = librosa.resample(array, orig_sr = sampling_rate, target_sr = 16000)

    feat_sample = feature_extractor(array, sampling_rate = 16000)
    feat_sample = torch.from_numpy(np.asarray(feat_sample["input_values"])[0]).unsqueeze(0).to(torch.float32)
    with torch.no_grad():
        pred = model(feat_sample.cuda()).logits.squeeze(0)
        pred = processor.decode(pred.cpu().detach().numpy())
        pred = remove_lonely_jamo(combine_jamo(pred))
        pred_list.append(pred) 

submission = pd.read_csv('sample_submission.csv')
submission['text'] = pd.Series(pred_list)
submission.to_csv('sample_submission.csv', index=False)