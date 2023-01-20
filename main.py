import torch
import numpy as np
import pandas as pd
import utils

import argparse

from knowledge.generate_knowledge import local_graph_expansion
from knowledge.generate_context import masked_sentence_score

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='csqa', choices=['csqa', 'csqa2'])
parser.add_argument('--model-type', type=str,
                    default='roberta-base', choices=['roberta-base'])
parser.add_argument("--save", type=bool, default=False,
                    help="Whether to save the trained model.")
parser.add_argument("--continue", type=bool, default=False,
                    help="Whether to save the trained model.")
args = parser.parse_args()

if __name__ == "__main__":
    concept_net = pd.read_csv('data/cpnet/conceptnet.en.csv', sep='\t', header=None)
    relation_text = utils.get_relationtext()
    ground_truths = pd.read_json('data/csqa/dev.json', lines=True)

    entities_list, query = [ground_truths.iloc[0]['entities']], ground_truths.iloc[0]['query']
    neighbors_list = []
    print("Start Local KG Expansion...")
    # K-hop expansion
    for i in range(3):
        entities, neighbors = local_graph_expansion(
            concept_net,
            entities_list[-1],
            query,
            use_prompt=False)
        entities_list.append(entities)
        neighbors_list.append(neighbors)
        print("Done, {}-th hop".format(i+1))

