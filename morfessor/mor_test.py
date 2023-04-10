import morfessor
if __name__ == '__main__':
    # io = morfessor.io.MorfessorIO()
    model = morfessor.baseline.AnnotatedCorpusEncoding()
    # model = io.read_binary_model_file('model.bin')
    #
    # words = ['words', 'segmenting', 'morfessor', 'unsupervised']
    #
    # for word in words:
    #     print(model.viterbi_segment(word))