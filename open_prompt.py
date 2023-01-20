from openprompt.pipeline_base import PromptForGeneration
from openprompt.prompts.generation_verbalizer import GenerationVerbalizer
from tqdm import tqdm
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ["negative", "positive"]


def prompt_setting(texts, question):
    plm, tokenizer, model_config, WrapperClass = load_plm("roberta-large", "roberta-large")
    dataset = []
    for i, x in enumerate(texts):
        dataset.append(InputExample(
            guid=i,
            text_a=x))

    promptTemplate = ManualTemplate(
        text='{"placeholder":"text_a"} is related to '+question+', which is {"mask"}',
        tokenizer=tokenizer)

    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words = {
            "negative": ["false"],
            "positive": ["true"],
        },
        tokenizer=tokenizer)

    promptModel = PromptForClassification(
        template=promptTemplate, plm=plm, verbalizer=promptVerbalizer)

    data_loader = PromptDataLoader(
        dataset=dataset, tokenizer=tokenizer, template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass)

    return promptModel, data_loader


def prompt_reasoning(texts, question, answer_candidates=None):
    if not answer_candidates:
        promptModel, data_loader = prompt_setting(texts, question)
    else:
        promptModel, data_loader = prompt_setting(texts, question, answer_candidates)
    print("Done, loading model and prompts")
    promptModel.to(device).eval()
    results = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            logits = torch.sigmoid(promptModel(batch.to(device)))
            results.append(logits)
            if i % 10 == 0:
                print("Done, {} of {}".format(i, len(data_loader)))
    return results


'''          
# making zero-shot inference using pretrained MLM with prompt
promptModel.eval()
with torch.no_grad():
    for batch in data_loader:
        logits = F.sigmoid(promptModel(batch))
        preds = torch.argmax(logits, dim=-1)
        print(logits)
        print(classes[preds])
    # predictions would be 1, 0 for classes 'positive', 'negative'
'''

