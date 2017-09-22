from keras.layers import LSTM, Embedding, Input, RepeatVector, TimeDistributed, Dense, Activation
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import RMSprop


MAX_LENGTH = 20

'''
Papers:
[1] Sequence to Sequence Learning with Neural Networks (http://arxiv.org/abs/1409.3215)
[2] Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation 
    (http://arxiv.org/abs/1406.1078)
[3] Neural Machine Translation by Jointly Learning to Align and Translate (http://arxiv.org/abs/1409.0473)
'''


def seq2seq(input_shape=(20,),
            hidden_units=256,
            source_words=1000,
            target_words=1000,
            target_length=25,
            decoder_layers=1):
    inputs = Input(shape=input_shape)

    # encoder
    embedding = Embedding(source_words, hidden_units, input_length=MAX_LENGTH)(inputs)
    encoding = LSTM(hidden_units, return_sequences=True)(embedding)
    encoding = LSTM(hidden_units)(encoding)
    encoding = RepeatVector(target_length)(encoding)

    # decoder
    decoder = LSTM(hidden_units, return_sequences=True)(encoding)
    for _ in range(decoder_layers):
        decoder = LSTM(hidden_units, return_sequences=True)(decoder)
    output = TimeDistributed(Dense(target_words))(decoder)
    output = Activation('softmax')(output)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss=categorical_crossentropy, optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])

    return model
