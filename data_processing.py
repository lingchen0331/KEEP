# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def clean_index(x):
    x = x.replace("text:", "")
    return float(x)

def clean_entities(x):
    x = x.split(']')
    x = [i.replace("[", '').split('|')[0] for i in x if i]
    return x
    
linked_entities = pd.read_csv('data/csqa/test_entities.txt', sep='\t', header=None)
ground_truths = pd.read_json('data/csqa/test_rand_split_no_answers.jsonl', lines=True)

linked_entities[0] = linked_entities[0].apply(clean_index)
linked_entities[0] = pd.to_numeric(linked_entities[0], errors='coerce')

linked_entities = linked_entities.sort_values(0)
linked_entities[3] = linked_entities[3].apply(clean_entities)

dfs = [{0: 364.0, 1: 'sent:0', 2: None,  3:None}]
for i in dfs:
    linked_entities = linked_entities.append(i, ignore_index=True)

linked_entities = linked_entities.sort_values(0)

questions, entities = list(linked_entities[2]), list(linked_entities[3])

ground_truths['query'] = questions
ground_truths['entities'] = entities
ground_truths = ground_truths.replace(to_replace='None', value=np.nan).dropna()
ground_truths = ground_truths.drop('question', axis=1)
ground_truths.to_json('data/csqa/test_csqa.json', orient='records', lines=True)