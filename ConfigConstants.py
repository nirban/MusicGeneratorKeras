class ConfigConstants:

    def __init__(self):
        self.MICROSECONDS_PER_MINUTE = 60000000
        self.time_per_time_slice = 0.02

        self.highest_note = 81  # A_5
        self.lowest_note = 33  # A_2
        self.input_dim = self.highest_note - self.lowest_note + 1  # number of notes in input
        self.output_dim = self.highest_note - self.lowest_note + 1  # number of notes in output

        self.x_seq_length = 50  # Piano roll matrix of dimention 50x49
        self.y_seq_length = 50  # Piano roll matrix of dimention 5

        print("Configurations Initialized........")

    def getMicroSecPerMins(self):
        return self.MICROSECONDS_PER_MINUTE

    def getTimePerTimeSlice(self):
        return self.time_per_time_slice

    def getHighestNote(self):
        return self.highest_note

    def getLowestNote(self):
        return self.lowest_note

    def getInputDim(self):
        return self.input_dim

    def getOutputDim(self):
        return self.output_dim

    def getXSeqLen(self):
        return self.x_seq_length

    def getYSeqLen(self):
        return self.y_seq_length

