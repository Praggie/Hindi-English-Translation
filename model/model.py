from keras.layers import LSTM, Embedding, Input


MAX_LENGTH = 20

'''
Papers:
[1] Sequence to Sequence Learning with Neural Networks (http://arxiv.org/abs/1409.3215)
[2] Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (http://arxiv.org/abs/1406.1078)
[3] Neural Machine Translation by Jointly Learning to Align and Translate (http://arxiv.org/abs/1409.0473)
'''


def seq2seq(input_shape, hidden_units, source_words, target_words, encoder_layers=1, decoder_layers=1):
    inputs = Input(shape=input_shape)

    # encoder
    embedding = Embedding(source_words, hidden_units, input_length=MAX_LENGTH)(inputs)
    encoding = LSTM(hidden_units)(embedding)
    for _ in range(encoder_layers):
        encoding = LSTM(hidden_units)(encoding)

    # decoder
    decoder = LSTM(hidden_units)(encoding)
