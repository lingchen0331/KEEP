import numpy as np
import pandas as pd
import torch
import os

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def question_prompt_generation(query: str) -> str:
    return query+" it is [MASK]."

def get_relationtext():
    merged_relations = [
    'antonym',
    'atlocation',
    'capableof',
    'causes',
    'createdby',
    'isa',
    'desires',
    'hassubevent',
    'partof',
    'hascontext',
    'hasproperty',
    'madeof',
    'notcapableof',
    'notdesires',
    'receivesaction',
    'relatedto',
    'usedfor',
]
    relation_text = [
    ' is the antonym of ',
    ' is at location of ',
    ' is capable of ',
    ' causes ',
    ' is created by ',
    ' is a kind of ',
    ' desires ',
    ' has subevent ',
    ' is part of ',
    ' has context ',
    ' has property ',
    ' is made of ',
    ' is not capable of ',
    ' does not desires ',
    ' is ',
    ' is related to ',
    ' is used for ',
]

    relation_text = dict(zip(merged_relations, relation_text))
    return relation_text

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
