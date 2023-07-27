import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from preprocess import make_bert_preprocess_model


def build_classifier_model(num_classes):
    class Classifier(tf.keras.Model):
        def __init__(self, num_classes):
            super(Classifier, self).__init__(name="prediction")
            self.encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True)
            self.dropout = tf.keras.layers.Dropout(0.1)
            self.dense = tf.keras.layers.Dense(num_classes)

        def call(self, preprocessed_text):
            encoder_outputs = self.encoder(preprocessed_text)
            pooled_output = encoder_outputs["pooled_output"]
            x = self.dropout(pooled_output)
            x = self.dense(x)
            return x

    model = Classifier(num_classes)
    return model


if __name__ == '__main__':
    test_preprocess_model = make_bert_preprocess_model(['my_input1', 'my_input2'])
    test_text = [np.array(['some random test sentence']),
                 np.array(['another sentence'])]
    text_preprocessed = test_preprocess_model(test_text)
    test_classifier_model = build_classifier_model(2)
    bert_raw_result = test_classifier_model(text_preprocessed)
    print(tf.sigmoid(bert_raw_result))
