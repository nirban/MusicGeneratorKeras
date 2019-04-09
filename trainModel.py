from loadData import *
from ConfigConstants import ConfigConstants

from Seq2SeqModel import *
import tensorflow as tf
import keras

#parralel Processing did not work
#config = tf.ConfigProto(device_count={"CPU": 8})
#keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


#Load Hyperparameters
cfgConst = ConfigConstants()
MICROSECONDS_PER_MINUTE = cfgConst.getMicroSecPerMins()
time_per_time_slice = cfgConst.getTimePerTimeSlice()
highest_note = cfgConst.getHighestNote()
lowest_note = cfgConst.getLowestNote()  # A_2
input_dim = cfgConst.getInputDim() # number of notes in input
output_dim = cfgConst.getOutputDim()  # number of notes in output

x_seq_length = cfgConst.getXSeqLen()  # Piano roll matrix of dimention 50x49
y_seq_length = cfgConst.getYSeqLen()

num_units = 128

#callBacks = [EarlyStopping(monitor='loss', patience=10, min_delta = 0.01 , verbose=0, mode='min'),TensorBoard(log_dir='output/graph_10Per', histogram_freq=1), History()]

callBacks = [EarlyStopping(monitor='val_loss', patience=10, min_delta = 0.01 , verbose=1, mode='min'),TensorBoard(log_dir='output/graph_10Per', histogram_freq=1), History()]

#Load Dataset
X_train, Y_train, X_val, Y_val = loadData()


#Create Model
model = createModel(num_units, input_dim, output_dim, x_seq_length, y_seq_length)

model.summary()

callBacks = [EarlyStopping(monitor='loss', patience=10, min_delta = 0.01 , verbose=0, mode='auto'),TensorBoard(log_dir='output/graph_10Per', histogram_freq=1), History()]

history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=64, epochs=120, shuffle='batch', callbacks=callBacks)

model.save('seq2seqModel_10PerDataset.h5')

'''
with h5py.File("seq2seqHistory.h5", "w") as f:
    f.create_dataset('loss', data=history.history['loss'])
    f.create_dataset('val_loss', data=history.history['val_loss'])


'''


