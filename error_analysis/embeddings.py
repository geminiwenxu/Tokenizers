import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

from resegment_explain.tokenization_bert_modified import ModifiedBertTokenizer

model_checkpoint = "bert-base-cased"
model = BertForSequenceClassification.from_pretrained(model_checkpoint, output_hidden_states=True)
modified_tokenizer = ModifiedBertTokenizer.from_pretrained(model_checkpoint, use_fast=True)
baseline_tokenizer = BertTokenizer.from_pretrained(model_checkpoint, use_fast=True)

if __name__ == '__main__':
    # sentence = "insensitive is how I would describe him."
    # sentence = "ungrateful"
    sentence = "ungrateful people forget what they are not grateful for."

    tok1 = modified_tokenizer(sentence, return_tensors='pt')
    tok2 = baseline_tokenizer(sentence, return_tensors='pt')
    print(tok1)
    print(modified_tokenizer.tokenize(sentence, return_tensors='pt'))
    print(tok2)
    print(baseline_tokenizer.tokenize(sentence, return_tensors='pt'))

    tok1_ids = [1, 2]
    tok2_ids = [1, 2, 3, 4]

    with torch.no_grad():
        modified_input = modified_tokenizer(sentence, return_tensors='pt')
        modified_output = model(**modified_input)
        baseline_input = baseline_tokenizer(sentence, return_tensors='pt')
        baseline_output = model(**baseline_input)

    # Only grab the last hidden state
    states1 = modified_output.hidden_states[-1].squeeze()
    print(states1.size())
    print(len(states1[0]))
    states2 = baseline_output.hidden_states[-1].squeeze()
    # Select the tokens that we're after corresponding to "New" and "York"
    embs1 = states1[[i for i in tok1_ids]]
    print(embs1.size())
    embs2 = states2[[i for i in tok2_ids]]

    avg1 = embs1.mean(axis=0)
    avg2 = embs2.mean(axis=0)
    cos_similarity = torch.cosine_similarity(avg1.reshape(1, -1), avg2.reshape(1, -1))
    print(cos_similarity)

    torch.save(avg1, 'method_2_3.pt')
    method_1 = torch.load('method_1_3.pt')
    method_2 = torch.load('method_2_3.pt')
    method_3 = torch.load('method_3_3.pt')
    cos_similarity = torch.cosine_similarity(method_1.reshape(1, -1), method_2.reshape(1, -1))
    print(cos_similarity)
    cos_similarity = torch.cosine_similarity(method_2.reshape(1, -1), method_3.reshape(1, -1))
    print(cos_similarity)
    cos_similarity = torch.cosine_similarity(method_1.reshape(1, -1), method_3.reshape(1, -1))
    print(cos_similarity)
