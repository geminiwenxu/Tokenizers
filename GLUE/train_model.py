epochs = 3
batch_size = 32
init_lr = 2e-5
from official.nlp import optimization

from preprocess import make_bert_preprocess_model
import tensorflow_addons as tfa
import tensorflow as tf
def get_configuration(glue_task):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if glue_task == 'glue/cola':
        metrics = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)
    else:
        metrics = tf.keras.metrics.SparseCategoricalAccuracy(
            'accuracy', dtype=tf.float32)

    return metrics, loss


print(f'Fine tuning {tfhub_handle_encoder} model')
bert_preprocess_model = make_bert_preprocess_model(sentence_features)

with strategy.scope():
    # metric have to be created inside the strategy scope
    metrics, loss = get_configuration(tfds_name)

    train_dataset, train_data_size = load_dataset_from_tfds(
        in_memory_ds, tfds_info, train_split, batch_size, bert_preprocess_model)
    steps_per_epoch = train_data_size // batch_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = num_train_steps // 10

    validation_dataset, validation_data_size = load_dataset_from_tfds(
        in_memory_ds, tfds_info, validation_split, batch_size,
        bert_preprocess_model)
    validation_steps = validation_data_size // batch_size

    classifier_model = build_classifier_model(num_classes)

    optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type='adamw')

    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    classifier_model.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=validation_steps)
