import morfessor


def morfesso_segmentation(word):
    io = morfessor.MorfessorIO()

    train_data = list(
        io.read_corpus_file('/Users/geminiwenxu/PycharmProjects/Tokenizers/morfessor-2.0.6/training_data'))

    model = morfessor.BaselineModel()  # Implements training of and segmenting with a Morfessor model.
    # lists of strings(finding phrases in sentences) o strings of characters(finding morpphes in words)

    model.load_data(train_data, count_modifier=lambda x: 1)
    # Load data to initialize the model for batch training.
    # data: iterator of (count, compound_atoms) tuples
    # freqthreshold: discard compounds that occur less than given times in the corpus
    # count_modifier: function for adjusting the counts of each compound
    # init_rand_split: if given, random split the word with int_rand_split as the probability for each split
    model.train_batch()  # train the model in the batch session
    segmentation = model.viterbi_segment(
        word)  # segmenting new words, Find optimal segmentation using the Viterbi algorithm.
    # Returns the most probable segmentation and its log-probability.
    return segmentation


if __name__ == '__main__':
    segmentation = morfesso_segmentation('colorful')
    print(segmentation)