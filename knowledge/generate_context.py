import torch
import numpy as np
from torch.nn import CrossEntropyLoss

def masked_sentence_score(model, tokenizer, sentence, device):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input.to(device), labels=labels.to(device)).loss
    return loss.cpu().detach()
'''
def generative_score(model, tokenizer, sentence, device):
    tokenize_input = tokenizer.encode(sentence)
    tensor_input = torch.tensor([tokenize_input]).to(device)
    loss = model(tensor_input, labels=tensor_input)[0]
    return loss.cpu().detach()
'''

def generative_score(model, tokenizer, sentence, device):
    tokenize_input = tokenizer.encode(sentence)
    tensor_input = torch.tensor([tokenize_input]).to(device)
    loss=model(tensor_input, labels=tensor_input)[0]
    return loss.cpu().detach()

'''
for i in ['Great Britain', 'North America', 'India', 'China']:
    print(masked_sentence_score(sentence='London is the capital of {}.'.format(i), model=model, tokenizer=tokenizer))
'''