# initialize the data collator, randomly masking 20% (default is 15%) of the tokens for the Masked Language
# Modeling (MLM) task
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

from approach_2.wp_tokenizer.model import build_model
from approach_2.wp_tokenizer.my_tokenizer import train_tokenizer
from approach_2.wp_tokenizer.prepare_dataset import prepare_dataset


def helper(ls_lists):
    """A help function to convert list of lists into a list.

    Args:
        list of lists.

    Returns:
        list[str]: Returns the list of morphemes.
    """
    ls = []
    for x in ls_lists:
        if isinstance(x, list):
            for i in range(len(x)):
                temp = x[i]
                ls.append(temp)
        else:
            ls.append(x)
    return ls


def training(data_train, data_test, special_tokens, vocab_size, max_length):
    special_tokens = helper(special_tokens)
    print("USED IN TRAINING", special_tokens)
    tokenizer = train_tokenizer(special_tokens)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.2
    )
    model_path = "greedy_tokenizer"
    model = build_model(vocab_size, max_length)
    training_args = TrainingArguments(
        output_dir=model_path,  # output directory to where save model checkpoint
        evaluation_strategy="steps",  # evaluate each `logging_steps` steps
        overwrite_output_dir=True,
        num_train_epochs=1,  # number of training epochs, feel free to tweak
        per_device_train_batch_size=10,  # the training batch size, put it as high as your GPU memory fits
        gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
        per_device_eval_batch_size=64,  # evaluation batch size
        logging_steps=1000,  # evaluate, log and save model checkpoints every 1000 step
        save_steps=1000,
        # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
        # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
    )
    # initialize the trainer and pass everything to it
    train_dataset, test_dataset = prepare_dataset(data_train, data_test)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    # train the model
    trainer.train()


if __name__ == '__main__':
    pass
