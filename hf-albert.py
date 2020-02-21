import os
from google_drive_downloader import GoogleDriveDownloader as gdd
import transformers
import torch
import tqdm

# ALBERT model files
# ref: https://qiita.com/mkt3/items/b41dcf0185e5873f5f75
ALBERT_FILES = {
    "./albert/spiece.model": "1EmvCNy9A3hs2awU_HIrloQepUnKILS9O",
    "./albert/config.json": "1AO9QT1X2G_tZjiCoLREqzptVbpFtiKLK",
    "./albert/pytorch_model.bin": "17H2macWy8gXfuv38BIB8nKofB5jgjksk"
}

for fname, fid in ALBERT_FILES.items():
    if os.path.exists(fname):
        print("already downloaded %s" % fname)
    else:
        gdd.download_file_from_google_drive(file_id=fid,
                                            dest_path=fname, unzip=False)

CONFIG_FILE = "./albert/config.json"
MODEL_DIR= "./albert"
TRAIN_FILE = "./train.tsv"

# flow:
# download model
# define network
# load data
# train

config = transformers.AlbertConfig.from_json_file(CONFIG_FILE)
config.num_labels = 1 # for regression
tokenizer = transformers.AlbertTokenizer.from_pretrained(MODEL_DIR, keep_accents=True)
model = transformers.AlbertForSequenceClassification.from_pretrained(MODEL_DIR, config=config)

# create examples
examples = []
with open(TRAIN_FILE) as f:
    for i, line in enumerate(f):
        row = line.strip().split()
        guid = "%s-%s" % ("train", i)
        text = row[3]
        score = row[2]
        examples.append(transformers.data.processors.utils.InputExample(
            guid=guid, text_a=text, label=score
        ))

# create features
features = []
max_length = config.max_length
pad_token = 0
pad_token_segment_id = 0

for ex_index, example in enumerate(examples):
    inputs = tokenizer.encode_plus(example.text_a, max_length=max_length)
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1] * len(input_ids)
    padding_length = max_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
    label = float(example.label)
    features.append(transformers.InputFeatures(
        input_ids=input_ids, attention_mask=attention_mask, 
        token_type_ids=token_type_ids, label=label))

device = "cpu"
if torch.cuda.is_available():
    device = "gpu"
model.to(device)
# train
sampler = torch.utils.data.RandomSampler(features)
dataloader = torch.utils.data.DataLoader(features, sampler=sampler, batch_size=8)
max_epoch = 50

optimizer = transformers.AdamW({}, lr=5e-5)
global_step = 0
tr_loss = 0.0
model.zero_grad()
train_iter = tqdm.trange(max_epoch, desc="Epoch")
for _ in train_iter:
    epoch_iter = tqdm.tqdm(dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iter):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
        'attention_mask': batch[1],
        'labels': batch[3]}
        outputs = model(**inputs)
        loss = outputs[0]
        loss.backward()

        tr_loss += loss.item()
        optimizer.setp()
        model.zero_grad()
        global_step += 1

