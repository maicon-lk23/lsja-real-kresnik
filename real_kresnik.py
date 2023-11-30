from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import librosa
import os
import numpy as np
import pandas as pd
import torch
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from jamo import h2j
import wandb
import random as rd
from audiomentations import *
from torch.cuda.amp import GradScaler, autocast
from utils import combine_jamo, preprocess_text

wandb.login()
run = wandb.init(project="wav2vec2 real kresnik Training")

processor = Wav2Vec2Processor.from_pretrained("42MARU/ko-spelling-wav2vec2-conformer-del-1s")
tokenizer = AutoTokenizer.from_pretrained("42MARU/ko-spelling-wav2vec2-conformer-del-1s")
model = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean").to("cuda")
model.lm_head = torch.nn.Linear(1024, len(tokenizer.vocab))

for name, params in model.named_parameters():
    if "feature_extractor" in name:
        params.requires_grad = False
    for i in range(24):
        if f"encoder.layers.{i}" in name:
            params.requires_grad = False

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

jamo_separater = lambda x : list(h2j(x))

def hangul_tokenizer(s):
    s = s.upper()
    s = preprocess_text(s)
    tokens = list(h2j(s))
    tokens = list(map(lambda x: tokenizer.vocab[x] if x in tokenizer.vocab else 1, tokens))     # 1: unk token
    return tokens


transforms = Compose([
    TimeStretch(min_rate=0.8, max_rate=1.3, p=0.5, leave_length_unchanged=False),
    # RoomSimulator(p=0.3),
    TimeMask(
        min_band_part=0.1,
        max_band_part=0.15,
        fade=True,
        p=0.3,
    ),
    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.5)
])

class TrainDataset(Dataset):
    BPATH = "C:\\Users\\Admin\\Downloads\\참가자용 데이터(학습데이터,설치파일,발표자료)\\Train Dataset_오디오 분야 과제"
    def __init__(self, indices, transforms=None):
        self.indices = indices
        self.label = pd.read_csv(os.path.join(self.BPATH, "train\\texts.csv"))["text"].tolist()
        self.label = list(map(hangul_tokenizer, self.label))
        self.transforms = transforms
    def __getitem__(self, idx):
        idx = self.indices[idx]
        array, sampling_rate = librosa.load(os.path.join(self.BPATH, f"denoise_dataset\\denoise_{idx}.wav"))
        array = librosa.resample(array, orig_sr = sampling_rate, target_sr = 16000)
        if self.transforms is not None:
            array = self.transforms(array, sample_rate=16000)
        feat_sample = feature_extractor(array, sampling_rate = 16000)
        feat_sample = feat_sample["input_values"][0]
        label = self.label[idx]
        return feat_sample if len(feat_sample) < int(5e5) else feat_sample[:int(5e5)], label
    def __len__(self):
        return len(self.indices)


class EvalDataset(Dataset):
    BPATH = "C:\\Users\\Admin\\Downloads\\참가자용 데이터(학습데이터,설치파일,발표자료)\\Train Dataset_오디오 분야 과제"
    def __init__(self, indices):
        self.indices = indices
        self.label = pd.read_csv(os.path.join(self.BPATH, "train\\texts.csv"))["text"].tolist()
    def __getitem__(self, idx):
        idx = self.indices[idx]
        array, sampling_rate = librosa.load(os.path.join(self.BPATH, f"denoise_dataset\\denoise_{idx}.wav"))
        array = librosa.resample(array, orig_sr = sampling_rate, target_sr = 16000)
        feat_sample = feature_extractor(array, sampling_rate = 16000)
        feat_sample = feat_sample["input_values"][0]
        label = self.label[idx]
        #feat_sample = torch.from_numpy(np.asarray(feat_sample["input_values"])[0]).unsqueeze(0).to(torch.float32)
        return feat_sample if len(feat_sample) < int(5e5) else feat_sample[:int(5e5)], label
    def __len__(self):
        return len(self.indices)


# Custom collator function
def collate_fn(batch):
    # Find the longest audio in the batch
    length_audio = tuple([len(audio) for audio, _ in batch])
    length_label = tuple([len(label) for _, label in batch])
    max_length_audio = max(length_audio)
    max_length_label = max(length_label)

    # Pad all audios to the max length
    batch_padded_audio = np.stack([np.pad(audio, (0, max_length_audio - len(audio)), mode='constant', constant_values=0) for audio, _ in batch])
    batch_padded_label = np.stack([np.pad(label, (tokenizer.pad_token_id, max_length_label - len(label)), mode='constant', constant_values=tokenizer.pad_token_id) for _, label in batch])
    
    # Convert to tensor
    batch_tensor_audio = torch.from_numpy(batch_padded_audio)
    batch_tensor_label = torch.from_numpy(batch_padded_label)

    return batch_tensor_audio, batch_tensor_label, length_audio, length_label

ds_size = 14000
indices = list(range(ds_size))
rd.seed(42)
rd.shuffle(indices)

train_size = int(len(indices) * 0.9)
train_indices = indices[:train_size]
eval_indices = indices[train_size:]
train_ds = TrainDataset(train_indices, transforms)
eval_ds = EvalDataset(eval_indices)

BATCH_SIZE=4

train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dl = DataLoader(dataset=eval_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists("./kresnik_model.pt"):
    model.load_state_dict(torch.load("./kresnik_model.pt"))

model.gradient_checkpointing_enable()
model = model.to(device)


cer_metric = torchmetrics.text.CharErrorRate()
loss_fn = torch.nn.CTCLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

epoch = 100
accumulation_step = 16
acc_list = []

scaler = GradScaler()
for e in range(epoch):
    model.train()
    losses = []
    for i, (array, label, len_arr, len_label) in tqdm(enumerate(train_dl), total=len(train_dl)):
        array, label = array.to(device), label.to(device)
        with autocast():
            pred = model(array).logits.log_softmax(dim=-1)
            pred = pred.permute(1, 0, 2)

            input_lengths = tuple([pred.shape[0]] * BATCH_SIZE)
        
            loss = loss_fn(pred, label, input_lengths=input_lengths, target_lengths=len_label) / accumulation_step
        scaler.scale(loss).backward()
        losses.append(loss.item() * accumulation_step)
        epoch_mean_loss = sum(losses) / len(losses)
        if (i + 1) % accumulation_step == 0 or i == len(train_dl) - 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        wandb.log({"train_loss_mean": epoch_mean_loss, "train_loss": losses[-1]})
    wandb.log({"lr": scheduler.get_last_lr()[0]})
    scheduler.step()

    with torch.no_grad():
        model.eval()
        trues, preds = [], []
        for array, label in tqdm(eval_ds):
            array = torch.from_numpy(array).unsqueeze(0).to(device)
            pred = model(array).logits.squeeze(0)
            pred = processor.decode(pred.cpu().detach().numpy())
            pred = combine_jamo(pred)

            preds.append(pred)
            trues.append(label)
    
        acc = 1 - cer_metric(list(map(preprocess_text, preds)), trues)
        acc_list.append(acc)
    if acc_list[-1] == max(acc_list):
        torch.save(model.state_dict(), "./kresnik_model.pt")
    context = {"epoch": e, "val_accuracy": acc, "train_loss": epoch_mean_loss}
    print(f"epoch : {epoch} - {list(map(lambda x: f'{x[0]} : {x[1]}', context.items()))}")
    wandb.log(context)
run.log_code()