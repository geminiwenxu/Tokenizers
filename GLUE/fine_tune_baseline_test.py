import os

DEV = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = DEV
import numpy as np
import yaml
from datasets import load_dataset, load_metric
from pkg_resources import resource_filename
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import enable_full_determinism
from copy import deepcopy
from transformers.integrations import TrainerCallback


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
epoch = 3
batch_size = 16
learning_rate = 0.001

# Enable random seed
enable_full_determinism(1337)
os.environ["CUDA_VISIBLE_DEVICES"] = DEV

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task = "mrpc"
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
        return tokenizer(examples[sentence1_key], truncation=True, padding="max_length")
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding="max_length")


encoded_dataset = dataset.map(preprocess_function, num_proc=26)

# Fine-tuning the model
num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epoch,
    weight_decay=0.01,
    logging_steps=10,
    logging_strategy="epoch",
    save_total_limit=1,
    save_strategy="epoch",
    metric_for_best_model=metric_name,
    load_best_model_at_end=True,
    greater_is_better=True
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)


validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"


def model_init():
    return BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


class CustomCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


if __name__ == '__main__':
    print("Baseline fine tune for", actual_task, "with LR and BS: ", learning_rate, batch_size)
    trainer.add_callback(CustomCallback(trainer))
    train = trainer.train()
    trainer.save_model(f"saved_model_{actual_task}")
    print("train log", train)
    trainer.evaluate()
    log_history = trainer.state.log_history
    print("log history", log_history)
    prediction = trainer.predict(encoded_dataset["test"])
    print(prediction)
