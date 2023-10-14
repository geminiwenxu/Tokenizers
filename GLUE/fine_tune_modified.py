import os
DEV ="1"
os.environ["CUDA_VISIBLE_DEVICES"] = DEV
import numpy as np
import yaml
from datasets import load_dataset, load_metric
from pkg_resources import resource_filename
from transformers import BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import enable_full_determinism

from resegment_explain.tokenization_bert_modified import ModifiedBertTokenizer


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
epoch = config['epoch']
batch_size = config['batch_size']
learning_rate = config['learning_rate']

# Enable random seed
enable_full_determinism(1337)
os.environ["CUDA_VISIBLE_DEVICES"] = DEV

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task = "stsb"
model_checkpoint = "bert-base-cased"
actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

# Preprocessing the data
tokenizer = ModifiedBertTokenizer.from_pretrained(model_checkpoint, use_fast=True)
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
    print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
    print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
    print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")


def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


encoded_dataset = dataset.map(preprocess_function, num_proc=26)

# Fine-tuning the model
num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
model = BertForSequenceClassification.from_pretrained(model_checkpoint,
                                                      num_labels=num_labels)
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epoch,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)


validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

if __name__ == '__main__':
    print("Modified fine tune for", actual_task, "with LR and BS: ", learning_rate, batch_size)
    trainer.train()
    trainer.evaluate()
    import pandas as pd
    pd.DataFrame(trainer.state.log_history)
    trainer.predict(test_dataset=encoded_dataset["test"])
