import os
import transformers
import torch
import tqdm

model_dir = "output"
DEV_FILE = "dev.tsv"

config = transformers.AutoConfig.from_pretrained(model_dir)
config.num_labels = 1
tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
model = transformers.AlbertForSequenceClassification.from_pretrained(model_dir, config=config)

# create examples
examples = []
with open(DEV_FILE) as f:
    for i, line in enumerate(f):
        row = line.strip().split()
        guid = "%s-%s" % ("dev", i)
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
model.to(device)

# predict
with torch.no_grad():
    sampler = torch.utils.data.RandomSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=8)
    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
        'attention_mask': batch[1],
        'labels':         batch[3]}
        outputs = model(**inputs)

        loss, logits = outputs[:2]

        import pdb; pdb.set_trace()
        pass
