from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector, Input
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, History, TensorBoard
from keras.optimizers import Adam
from keras.models import Model



def createModel(num_units, input_dim, output_dim, x_seq_length, y_seq_length):

    # encoder
    model = Sequential()
    model.add(LSTM(input_dim=input_dim, output_dim=num_units, activation='tanh', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(num_units, activation='tanh'))
    model.add(Dense(num_units, activation='sigmoid'))

    # decoder
    model.add(RepeatVector(y_seq_length))
    #num_layers = 2

    model.add(LSTM(num_units, activation='tanh', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(num_units, activation='tanh', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(TimeDistributed(Dense(output_dim, activation='softmax')))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def createModelEncDec(num_units=128, input_dim=49, output_dim=49, x_seq_length=50, y_seq_length=50):
    
    #encoder
    inputs1 = Input(shape=(50,49))
    lstm1 = LSTM(num_units, return_sequences=True)(inputs1)
    batchNorm = BatchNormalization()(lstm1)
    dropout = Dropout(0.3)(batchNorm)
    denseor = Dense(num_units, activation='sigmoid')(dropout)
    lstm2, state_h, state_c = LSTM(num_units, return_state=True)(denseor)
    encoder_states = [state_h, state_c]
    
    #decoder
    decIn = Input(shape=(50,49))
    #lstm3 = LSTM(num_units, return_sequences=True)(inputs1)
    lstm3 = LSTM(num_units, return_sequences=True, return_state=True)
    lstmOut, _, _ = lstm3(decIn, initial_state=encoder_states)
    outDense = TimeDistributed(Dense(output_dim, activation='softmax'))
    decoder_outputs = outDense(lstmOut)
    
    model = Model(inputs=[inputs1, decIn] , outputs=[decoder_outputs])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
