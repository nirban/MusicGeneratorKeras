import math, os, glob, pickle
import numpy as np
import pandas as pd
import progressbar

from mido import MidiFile, Message, MetaMessage, MidiTrack

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.utils import shuffle

from ConfigConstants import ConfigConstants


class MIDIDataProcess:

    def __init__(self):
        cfgConst = ConfigConstants()
        self.MICROSECONDS_PER_MINUTE = cfgConst.getMicroSecPerMins()
        self.time_per_time_slice = cfgConst.getTimePerTimeSlice()
        self.highest_note = cfgConst.getHighestNote()
        self.lowest_note = cfgConst.getLowestNote()  # A_2
        self.input_dim = cfgConst.getInputDim() # number of notes in input
        self.output_dim = cfgConst.getOutputDim()  # number of notes in output

        self.x_seq_length = cfgConst.getXSeqLen()  # Piano roll matrix of dimention 50x49
        self.y_seq_length = cfgConst.getYSeqLen()

        print("MIDI Data parameters configured......")

    def midiToMatrix(self, filepath):

        midi_data = MidiFile(filepath)
        resolution = midi_data.ticks_per_beat

        #print("resolution : ", resolution)

        set_tempo_events = [x for t in midi_data.tracks for x in t if str(x.type) == 'set_tempo']

        tempo = self.MICROSECONDS_PER_MINUTE / set_tempo_events[0].tempo

        #print("tempo : ", tempo)

        ticks_per_time_slice = 1.0 * (resolution * tempo * self.time_per_time_slice) / 60

        #print("ticks_per_time_slice : ", ticks_per_time_slice)

        # Get maximum ticks across all tracks
        total_ticks = 0
        for t in midi_data.tracks:
            # since ticks represent delta times we need a cumulative sum to get the total ticks in that track
            sum_ticks = 0
            for e in t:
                if str(e.type) == 'note_on' or str(e.type) == 'note_off' or str(e.type) == 'end_of_track':
                    sum_ticks += e.time

            if sum_ticks > total_ticks:
                total_ticks = sum_ticks

        
        #print("total_ticks : ", total_ticks)

        time_slices = int(math.ceil(total_ticks / ticks_per_time_slice))

        #print("time_slices : ", time_slices)

        matrixPiano = np.zeros((self.input_dim, time_slices), dtype=int)  #a Matrix

        note_states = {}
        for track in midi_data.tracks:
            total_ticks = 0
            for event in track:
                if str(event.type) == 'note_on' and event.velocity > 0:
                    total_ticks += event.time
                    time_slice_idx = int(total_ticks / ticks_per_time_slice)

                    if event.note <= self.highest_note and event.note >= self.lowest_note:
                        note_idx = event.note - self.lowest_note
                        matrixPiano[note_idx][time_slice_idx] = 1
                        note_states[note_idx] = time_slice_idx

                elif str(event.type) == 'note_off' or (str(event.type) == 'note_on' and event.velocity == 0):
                    note_idx = event.note - self.lowest_note
                    total_ticks += event.time
                    time_slice_idx = int(total_ticks / ticks_per_time_slice)

                    if note_idx in note_states:
                        last_time_slice_index = note_states[note_idx]
                        matrixPiano[note_idx][last_time_slice_index:time_slice_idx] = 1
                        del note_states[note_idx]

        return matrixPiano.T

    def MatrixToMidi(self, matrixPiano, filepath):
        # ensure that resolution is an integer
        ticks_per_time_slice = 1 
        tempo = 1 / self.time_per_time_slice
        resolution = 60 * ticks_per_time_slice / (tempo * self.time_per_time_slice)

        mid = MidiFile(ticks_per_beat=int(resolution))
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('set_tempo', tempo=int(self.MICROSECONDS_PER_MINUTE / tempo), time=0))

        current_state = np.zeros(self.input_dim)

        index_of_last_event = 0

        for slice_index, time_slice in enumerate(np.concatenate((matrixPiano, np.zeros((1, self.input_dim))), axis=0)):
            note_changes = time_slice - current_state

            for note_idx, note in enumerate(note_changes):
                if note == 1:
                    note_event = Message('note_on', time=(slice_index - index_of_last_event) * ticks_per_time_slice,
                                         velocity=65, note=note_idx + self.lowest_note)
                    track.append(note_event)
                    index_of_last_event = slice_index
                elif note == -1:
                    note_event = Message('note_off', time=(slice_index - index_of_last_event) * ticks_per_time_slice,
                                         velocity=65, note=note_idx + self.lowest_note)
                    track.append(note_event)
                    index_of_last_event = slice_index

            current_state = time_slice

        eot = MetaMessage('end_of_track', time=1)
        track.append(eot)

        mid.save(filepath)

    def createXYDataset(self, matrixPianoData, x_seq_length, y_seq_length):
        x = []
        y = []
        #X = np.ones(shape=(1, x_seq_length, self.input_dim))
        #Y = np.ones(shape=(1, y_seq_length, self.input_dim))
        print("\nCreating Dataset for matrix list of length.... \n", len(matrixPianoData))

        prog = 0
        bar = progressbar.ProgressBar(max_value=len(matrixPianoData))
        for i, matrixPiano in enumerate(matrixPianoData):
            pos = 0
            while pos + x_seq_length + y_seq_length < matrixPiano.shape[0]:
                #X = np.vstack((X, [matrixPiano[pos:pos + x_seq_length]]))
                #Y = np.vstack((Y, [matrixPiano[pos + x_seq_length: pos + x_seq_length + y_seq_length]]))
                x.append(matrixPiano[pos:pos + x_seq_length])
                y.append(matrixPiano[pos + x_seq_length: pos + x_seq_length + y_seq_length])
                #print(X[1:,:,:].shape, Y[1:,:,:].shape )
                pos += x_seq_length
            prog = prog + 1
            bar.update(prog)
        
        matrixPianoData = None
        x = np.array(x)
        y = np.array(y)
        #X, Y = shuffle(x, y) #Causing MemoryError: exception
        print("Dataset Created with shape X and Y....\n", x.shape, y.shape )

        return x, y

    def getPandasData(self, dataFrames, directory):
        matrixPianoData = []
        prog = 0
        bar = progressbar.ProgressBar(max_value=len(dataFrames))
        for file in dataFrames:
            filepath = directory + "/" + file
            matrixPiano = self.midiToMatrix(filepath)
            matrixPianoData.append(matrixPiano)
            prog += 1
            bar.update(prog)

        return matrixPianoData

    def getData(self, data_dir):
        matrixPianoData = []
        prog = 0
        bar = progressbar.ProgressBar(max_value=len([name for name in os.listdir(data_dir)]))
        for file in os.listdir(data_dir):
            filepath = data_dir + "/" + file
            matrixPiano = self.midiToMatrix(filepath)
            matrixPianoData.append(matrixPiano)
            prog += 1
            bar.update(prog)

        return matrixPianoData

