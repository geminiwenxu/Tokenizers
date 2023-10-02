import os

import numpy as np
from datasets import load_dataset, load_metric
from optuna.samplers import TPESampler
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import enable_full_determinism

# Enable random seed
enable_full_determinism(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task = "wnli"
model_checkpoint = "bert-base-cased"
actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

# Preprocessing the data
tokenizer = BertTokenizer.from_pretrained(model_checkpoint, use_fast=True)
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

num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
training_args = TrainingArguments("test", num_train_epochs=20, evaluation_strategy="steps", eval_steps=500,
                                  disable_tqdm=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)


validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"


# Define the hyperparameter search space
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128])
    }


# Define a model_init function and pass it to the trainer
def model_init():
    return BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


# Create a Trainer with the model_init function, training arguments, training and test datasets and evaluation function
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=encoded_dataset["train"],  # .shard(index=1, num_shards=10),  # get 1/10 of the dataset
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


def my_objective(metrics):
    # print("attention", metrics)
    # return metrics['eval_f1']
    # return metrics['eval_matthews_correlation']
    # return metrics['eval_pearson']
    return metrics['eval_accuracy']


if __name__ == '__main__':
    print("Actual task of Baseline", actual_task)
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=20,
        compute_objective=my_objective,
        sampler=TPESampler(seed=42)
    )
    print("The best trail: ", best_trial)
