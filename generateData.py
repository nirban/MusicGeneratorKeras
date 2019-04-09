import pandas as pd
import numpy as np
from MIDIDataProcess import MIDIDataProcess


def generateData(x_seq_length=50, y_seq_length=50):

    mdp = MIDIDataProcess()

    MaestroDiataCSV = 'maestro-v1.0.0/maestro-v1.0.0.csv'

    df = pd.read_csv(MaestroDiataCSV)
    trainData = df[df.split == 'train'].midi_filename
    trainLen = len(trainData)
    testData = df[df.split == 'test'].midi_filename
    testLen = len(testData)
    validationData = df[df.split == 'validation'].midi_filename
    validationLen = len(validationData)

    trainDataList = []
    for i in trainData:
        trainDataList.append(i)

    testDataList = []
    for i in testData:
        testDataList.append(i)

    validationDataList = []
    for i in validationData:
        validationDataList.append(i)


    pianoRollDataTrain = mdp.getPandasData(trainDataList[95*9:], 'maestro-v1.0.0')
    X_train, Y_train = mdp.createXYDataset(pianoRollDataTrain, x_seq_length=x_seq_length, y_seq_length=y_seq_length)
    X_train = X_train.astype(np.bool)
    Y_train = Y_train.astype(np.bool)
    pianoRollDataTrain = None
    print("\nTraining data processing completed....\nShape of data is : ", X_train.shape)

    pianoRollDataTest = mdp.getPandasData(testDataList[12*9:], 'maestro-v1.0.0')
    X_test, Y_test = mdp.createXYDataset(pianoRollDataTest, x_seq_length=x_seq_length, y_seq_length=y_seq_length)
    X_test = X_test.astype(np.bool)
    Y_test = Y_test.astype(np.bool)
    pianoRollDataTest = None
    print("\nTest data processing completed....\nShape of data is : ", X_test.shape)

    pianoRollDataValidate = mdp.getPandasData(validationDataList[10*9:], 'maestro-v1.0.0')
    X_val, Y_val = mdp.createXYDataset(pianoRollDataValidate, x_seq_length=x_seq_length, y_seq_length=y_seq_length)
    X_val = X_val.astype(np.bool)
    Y_val = Y_val.astype(np.bool)
    pianoRollDataValidate = None
    print("\nValidation data processing completed....\nShape of data is : ", X_val.shape)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test
