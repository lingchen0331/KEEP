import random
import numpy as np
import pandas as pd
import utils
import torch
import time

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM
from open_prompt import prompt_reasoning
from sentence_similarity import sentence_embeddings
from generate_context import masked_sentence_score, generative_score

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

relation_text = utils.get_relationtext()
model_name = 'gpt2-large'
with torch.no_grad():
    #model = AutoModelWithLMHead.from_pretrained(model_name).to(device)
    #model = AutoModelForMaskedLM.from_pretrained('../models/finetuned_csqa_roberta_large/model_1103').to(device)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)


def neighbors_pick(concept_net: pd.DataFrame, entity: str) -> pd.DataFrame:
    """ Retrieving all concepts that are neighbors of the passed-in concept"""
    entity = entity.replace(' ', '_')
    return concept_net.loc[concept_net[1] == entity]


def bfs(concept_net: pd.DataFrame, entities: list[str]):
    """ Iterate the neighbors and find the most likely entities"""
    df = pd.DataFrame()
    for entity in entities:
        df = pd.concat([df, neighbors_pick(concept_net, entity)])

    sentences, answer_candidates = [], []
    for index, row in df.iterrows():
        temp = ''.join([row[1], relation_text[row[0]], row[2]])
        sentences.append(temp)
        answer_candidates.append(row[2])
    return df, sentences, answer_candidates


def embedding_comparison(sentences, question, tokenizer, model, k_hop=10):
    sentence_embs = sentence_embeddings(sentences, tokenizer, model, device)
    question_embs = sentence_embeddings(question, tokenizer, model, device)
    scores = torch.tensor([cos(question_embs, k.unsqueeze(0)) for k in sentence_embs])
    #scores = torch.mm(sentence_embs, question_embs.T)
    top_entities = torch.topk(scores, k_hop)
    return top_entities


def prompt_prob_comparison(sentences, question, answer_candidates=None, k_hop=10):
    """Using Prompt-based way to select top answer candidates"""
    results = prompt_reasoning(sentences, question, answer_candidates)
    results = torch.tensor([i.cpu()[:, 1].item() for i in results])
    top_entities = torch.topk(results, k_hop)
    return top_entities


def masked_sentence_prob_comparison(sentences, question, answer_candidates=None, k_hop=10, masked=False):
    results, _sentences = [], []
    for i, x in enumerate(sentences):
        if not masked:
            score = generative_score(
            sentence=question+' {}, because {}.'.format(
                answer_candidates[i], x),
            model=model,
            tokenizer=tokenizer,
            device=device)
        else:
            score = masked_sentence_score(
                sentence=question + ' {}, because {}.'.format(
                    answer_candidates[i], x),
                model=model,
                tokenizer=tokenizer,
                device=device)
        #print("The score of concept: {} in {} is {}".format(answer_candidates[i], x, score))
        results.append(score)
    results = -torch.tensor(results)
    top_entities = torch.topk(results, k_hop)
    return top_entities


def local_graph_expansion(concept_net, entities, query, use_prompt=True):
    neighbors, sentences, answer_candidates = bfs(concept_net, entities)
    if use_prompt:
        results = masked_sentence_prob_comparison(sentences, query, answer_candidates)
    else:
        results = embedding_comparison(sentences, query, tokenizer, model)
    neighbors = neighbors.iloc[results.indices]
    entities = list(set(neighbors[2]))

    entitiy_scores = list(zip(entities, results.values.tolist()))
    return entities, neighbors, entitiy_scores


if __name__ == '__main__':
    concept_net = pd.read_csv('../data/cpnet/conceptnet.en.csv', sep='\t', header=None)
    ground_truths = pd.read_json('../data/csqa/dev.json', lines=True)

    query = list(ground_truths['query'])
    answer = list(ground_truths['answer'])
    scores = 0
    for i, x in enumerate(query):
        temp = x + ' ' + answer[i]
        score = generative_score(
            sentence=temp,
            model=model,
            tokenizer=tokenizer,
            device=device)
        print(temp + ": " + str(score.item()))
        scores += score.item()
    print("Average score is "+str(scores/len(query)))


    #column_names = ["Query", "Ground Truths", "Answer 1", "Answer 2", "Answer 3", "Answer 4", "Answer 5", "Top 1 Acc", "Top 3 Acc", "Top 5 Acc"]
    #df = pd.DataFrame(columns=column_names)
    query_list, truth_list = [], []
    answer_1_list, answer_2_list, answer_3_list, answer_4_list, answer_5_list = [], [], [], [], []
    for k in range(0, 1):
        entities_list, query = [ground_truths.iloc[k]['entities']], \
                                ground_truths.iloc[k]['query']
        print(query)
        neighbors_list = []
        scores_list = []

        for i in range(2):
            try:
                entities, neighbors, results = local_graph_expansion(
                    concept_net,
                    entities_list[-1],
                    query,
                    use_prompt=True)
                entities_list.append(entities)
                scores_list.extend(results)
                print("Done, {}-th hop".format(i+1))
            except:
                continue

        scores_list.sort(key=lambda x: x[1], reverse=True)

        query_list.append(query)
        truth_list.append(ground_truths.iloc[k]['query'])

        '''
        try:
            temp_data = [k, query] + scores_list[:5]
        except:
            temp_data = [k, query] + [None]*5
        temp_data += [None]*3

        print(temp_data)
        with open("/home/ds/cling/cling/prompt_graph_reasoning/test.csv", "a") as myfile:
            myfile.write(";".join([str(i) for i in temp_data]))
            myfile.write("\n")
        '''
        '''
        try:
            answer_1_list.append(scores_list[0])
            answer_2_list.append(scores_list[1])
            answer_3_list.append(scores_list[2])
            answer_4_list.append(scores_list[3])
            answer_5_list.append(scores_list[4])
        except:
            answer_1_list.append(None)
            answer_2_list.append(None)
            answer_3_list.append(None)
            answer_4_list.append(None)
            answer_5_list.append(None)


    answer_cands = list(ground_truths.cands)[:1000]
    dict = {'Query': query_list,
            'Ground Truths': answer_cands,
            'Answer 1': answer_1_list,
            'Answer 2': answer_2_list,
            'Answer 3': answer_3_list,
            'Answer 4': answer_4_list,
            'Answer 5': answer_5_list,
            'Top_1': [None] * len(answer_1_list),
            'Top_3': [None] * len(answer_1_list),
            'Top_5': [None] * len(answer_1_list),
            }
    df = pd.DataFrame(dict)
    df.to_csv('/home/ds/cling/cling/prompt_graph_reasoning/test.csv')
    
    '''
    #_start_time = time.time()
    #neighbors, sentences = bfs(concept_net, ground_truths.iloc[0]['entities'])
    #results = prompt_reasoning(sentences, query)
    #print("--- %s seconds ---" % (time.time() - _start_time))
    #neighbors, sentences = bfs(concept_net, ground_truths.iloc[0]['entities'])
    #results = prompt_reasoning(sentences, ground_truths.iloc[0]['query'])

'''
def prompt_format(prompt_path, keywords, query):
    with open(prompt_path) as f:
        context_string = f.read().strip('\n')
    if keywords is not None:
        n = np.random.choice(range(1, len(keywords)+1))      # number of keywords
        keywords = random.sample(keywords, n)                # subset of keywords
        context_string = context_string.replace('{keywords}', ', '.join(keywords))
    if query is not None:
        context_string = context_string.replace('{question}', query)
    return context_string
'''