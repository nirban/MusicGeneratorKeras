from __future__ import print_function
import numpy as np
import sys

from keras.models import load_model, Model

from MIDIDataGen import MIDIDataGen
from ConfigConstants import ConfigConstants
from MIDIDataProcess import MIDIDataProcess


def generateMusic(modelFile='seq2seqModel_10PerDataset.h5', outFile='out_test.midi'):
    #intialize utility classes
    md = MIDIDataProcess()
    gmd = MIDIDataGen()
    cfgConst = ConfigConstants()
    
    #load the pre trained model
    model = load_model(modelFile)

    testMatrix = [md.midiToMatrix('dataTest/3.midi')]

    #generate a random piano roll matrix which has no musical data
    randTest = [np.random.choice(a=[True, False], size=(5000, 49), p=[0.35, 1-0.35])]

    testInput = gmd.matrixToModelInputs(randTest, cfgConst.y_seq_length)

    for i,song in enumerate(testInput):
        netOutput = model.predict(song)
        netRoll = gmd.modelPredictToMatrixPiano(netOutput)
        md.MatrixToMidi(netRoll[:5000,:], outFile)


if __name__ == '__main__':
    #generateMusic()
    print(len(sys.argv), sys.argv)
    if len(sys.argv) < 2:
        print("\ngenerating Music in deafult....\n")
        generateMusic()
