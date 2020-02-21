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
OUTPUT_DIR = "output"

config = transformers.AlbertConfig.from_json_file(CONFIG_FILE)
config.num_labels = 1 # for regression
tokenizer = transformers.AlbertTokenizer.from_pretrained(MODEL_DIR, keep_accents=True)
model = transformers.AlbertForSequenceClassification.from_pretrained(MODEL_DIR, config=config)

# freeze lower layers
# ref: https://github.com/nekoumei/DocumentClassificationUsingBERT-Japanese
# 1. まず全部を、勾配計算Falseにしてしまう
for name, param in model.named_parameters():
    param.requires_grad = False
# 2. 最後のBertLayerモジュールを勾配計算ありに変更
for name, param in model.albert.encoder.albert_layer_groups[-1].named_parameters():
    param.requires_grad = True
# 3. 識別器を勾配計算ありに変更
for name, param in model.classifier.named_parameters():
    param.requires_grad = True

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

# create dataset
all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
dataset = torch.utils.data.TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
model.to(device)
# train
sampler = torch.utils.data.RandomSampler(dataset)
dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=8)
max_epoch = 100

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=5e-5)
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
        optimizer.step()
        model.zero_grad()
        global_step += 1

# save result
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
