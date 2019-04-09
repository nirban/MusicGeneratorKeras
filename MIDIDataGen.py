import numpy as np
import os
from mido import MidiFile, Message, MetaMessage, MidiTrack

from ConfigConstants import ConfigConstants

class MIDIDataGen:
    def __init__(self):
        cfgConst = ConfigConstants()
        self.MICROSECONDS_PER_MINUTE = cfgConst.getMicroSecPerMins()
        self.time_per_time_slice = cfgConst.getTimePerTimeSlice()
        self.highest_note = cfgConst.getHighestNote()
        self.lowest_note = cfgConst.getLowestNote()  # A_2
        self.input_dim = cfgConst.getInputDim()  # number of notes in input
        self.output_dim = cfgConst.getOutputDim()  # number of notes in output

        self.x_seq_length = cfgConst.getXSeqLen()  # Piano roll matrix of dimention 50x49
        self.y_seq_length = cfgConst.getYSeqLen()

        print("MIDI Data parameters configured......")

    def matrixToModelInputs(self, matrixPianoData, seq_length):
        #matrixPianoData is a single list of the piano roll matrix
        #returns many piano rolls of seq_length
        x_test = []
        for i, matrixPiano in enumerate(matrixPianoData):
            print("matrixPiano.shape :", matrixPiano.shape)
            x = []
            pos = 0
            while pos + seq_length < matrixPiano.shape[0]:
                x.append(matrixPiano[pos:pos + seq_length])
                pos += 1
            x_test.append(np.array(x))

        print("x_test shape", np.array(x_test).shape)

        return np.array(x_test)

    def modelPredictToMatrixPiano(self, output, threshold=0.1):
        #returns a piano roll matrix based on the prediction done by the model,
        #the thrshold is used for the softmax probabilities
        matrixPiano = []
        for seq_out in output:
            for time_slice in seq_out:
                idx = [i for i, t in enumerate(time_slice) if t > threshold]
                matrixPianoSlice = np.zeros(time_slice.shape)
                matrixPianoSlice[idx] = 1
                matrixPiano.append(matrixPianoSlice)

        return np.array(matrixPiano)
