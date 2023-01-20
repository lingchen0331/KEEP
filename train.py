# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import utils
import torch
import time

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
from transformers import Trainer, TrainingArguments


def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples], truncation=True)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = 'roberta-large'

model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/csqa/csqa_logical_query_sentences_train.txt",
    block_size=100,
)
eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/csqa/csqa_logical_sentences_val.txt",
    block_size=100,
)

training_args = TrainingArguments(
    output_dir="./models",
    overwrite_output_dir=True,
    num_train_epochs=2,
    learning_rate=1e-5,
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,
    save_steps=10000,
    save_total_limit=2
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
trainer.save_model('./models/finetuned_csqa_roberta_large/model_1103_query')