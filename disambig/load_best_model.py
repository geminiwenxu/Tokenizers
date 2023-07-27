import tensorflow as tf

if __name__ == '__main__':
    folder_path = "/Users/geminiwenxu/PycharmProjects/Tokenizers/disambig/bigsense/en/best_model"
    model_path = "/Users/geminiwenxu/PycharmProjects/Tokenizers/disambig/bigsense/en/best_model/saved_model.pb"
    variables_path = "/Users/geminiwenxu/PycharmProjects/Tokenizers/disambig/bigsense/en/best_model/variables"
    # model = tf.saved_model.load(folder_path)
    # loaded_model = tf.keras.saving.load_model(folder_path)
    pretrained_model ="/Users/geminiwenxu/PycharmProjects/Tokenizers/xlmr-base"
    model = tf.saved_model.load(pretrained_model)

