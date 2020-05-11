import os
import transformers
import torch
import tqdm

model_dir = "output"

config = transformers.AutoConfig.from_pretrained(model_dir)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
model = transformers.AutoModel.from_pretrained(model_dir)

import pdb; pdb.set_trace()
pass
