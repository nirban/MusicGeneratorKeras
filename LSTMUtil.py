import numpy as np

from ConfigConstants import ConfigConstants
from MIDIDataProcess import MIDIDataProcess


class LSTMUtil:

    def initialize_parameters(self, n_a, n_x, n_y):
        """
        Initialize parameters with small random values

        Returns:
        parameters -- python dictionary containing:

                Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
        """
        np.random.seed(1)

        Wf = np.random.randn(n_a, n_a + n_x) * 0.01
        bf = np.random.randn(n_a, 1) * 0.01
        Wi = np.random.randn(n_a, n_a + n_x) * 0.01
        bi = np.random.randn(n_a, 1) * 0.01
        Wc = np.random.randn(n_a, n_a + n_x) * 0.01
        bc = np.random.randn(n_a, 1) * 0.01
        Wo = np.random.randn(n_a, n_a + n_x) * 0.01
        bo = np.random.randn(n_a, 1) * 0.01
        Wy = np.random.randn(n_y, n_a) * 0.01
        by = np.random.randn(n_y, 1) * 0.01

        parameters = {'Wf': Wf, 'bf': bf, 'Wi': Wi, 'bi': bi, 'Wc': Wc, 'bc': bc, 'Wo': Wo, 'bo': bo, 'Wy': Wy, 'by': by}

        return parameters

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def lstm_cell_forward(self, xt, a_prev, c_prev, parameters):

        """
        Implement a single forward step of the LSTM-cell as described in Figure (4)

        Arguments:
        xt -- your input data at timestep "t", numpy array of shape (n_x, m).
        a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
        c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
        parameters -- python dictionary containing:
                Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

        Returns:
        a_next -- next hidden state, of shape (n_a, m)
        c_next -- next memory state, of shape (n_a, m)
        yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
        cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)

        Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
              c stands for the memory value
        """

        # Retrieve parameters from "parameters"
        Wf = parameters["Wf"]
        bf = parameters["bf"]
        Wi = parameters["Wi"]
        bi = parameters["bi"]
        Wc = parameters["Wc"]
        bc = parameters["bc"]
        Wo = parameters["Wo"]
        bo = parameters["bo"]
        Wy = parameters["Wy"]
        by = parameters["by"]

        # Retrieve dimensions from shapes of xt and Wy
        m, n_x = xt.shape
        n_y, n_a = Wy.shape

        # Concatenate a_prev and xt
        concat = np.zeros(shape=(m, n_a + n_x))
        print("concat shape : ", concat.shape)
        print("a_prev shape : ", a_prev.shape)
        print("xt shape : ", xt.shape)
        concat[:, n_a:] = xt
        concat[:, :n_a] = a_prev

        # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
        print("Wf and bf : ", Wf.shape, bf.shape)
        ft = self.sigmoid(np.dot(Wf, concat) + bf)
        it = self.sigmoid(np.dot(Wi, concat) + bi)
        cct = np.tanh(np.dot(Wc, concat) + bc)
        c_next = ft * c_prev + it * cct
        ot = self.sigmoid(np.dot(Wo, concat) + bo)
        a_next = ot * np.tanh(c_next)

        # Compute prediction of the LSTM cell
        yt_pred = self.softmax(np.dot(Wy, a_next) + by)

        # store values needed for backward propagation in cache
        cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

        return a_next, c_next, yt_pred, cache

    def lstm_forward(self, x, a0, parameters):
        """
        Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (3).

        Arguments:
        x -- Input data for every time-step, of shape (m, T_x, n_x).
        a0 -- Initial hidden state, of shape (m, n_a)
        parameters -- python dictionary containing:
                            Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                            bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                            Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                            bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                            Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                            bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                            Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                            bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                            Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

        Returns:
        a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
        y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
        caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
        """

        # Initialize "caches", which will track the list of all the caches
        caches = []

        ### START CODE HERE ###
        # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
        m, T_x, n_x = x.shape
        n_y, n_a = parameters['Wy'].shape

        # initialize "a", "c" and "y" with zeros (≈3 lines)
        a = np.zeros(shape=(m, T_x, n_a))
        c = np.zeros(shape=(m, T_x, n_a))
        y = np.zeros(shape=(m, T_x, n_y))

        # Initialize a_next and c_next (≈2 lines)
        a_next = a0
        c_next = np.zeros_like(a_next)

        # loop over all time-steps
        for t in range(T_x):
            # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
            a_next, c_next, yt, cache = self.lstm_cell_forward(x[:, t, :], a_next, c_next, parameters)
            # Save the value of the new "next" hidden state in a (≈1 line)
            a[:, t, :] = a_next
            # Save the value of the prediction in y (≈1 line)
            y[:, t, :] = yt
            # Save the value of the next cell state (≈1 line)
            c[:, t, :] = c_next
            # Append the cache into caches (≈1 line)
            caches.append(cache)

        ### END CODE HERE ###

        # store values needed for backward propagation in cache
        caches = (caches, x)

        return a, y, c, caches

    def lstm_cell_backward(self, da_next, dc_next, cache):
        """
        Implement the backward pass for the LSTM-cell (single time-step).

        Arguments:
        da_next -- Gradients of next hidden state, of shape (n_a, m)
        dc_next -- Gradients of next cell state, of shape (n_a, m)
        cache -- cache storing information from the forward pass

        Returns:
        gradients -- python dictionary containing:
                            dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                            da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                            dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                            dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                            dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                            dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                            dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                            dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                            dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                            dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                            dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
        """

        # Retrieve information from "cache"
        (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

        ### START CODE HERE ###
        # Retrieve dimensions from xt's and a_next's shape (≈2 lines)
        n_x, m = xt.shape
        n_a, m = a_next.shape

        # Retrieve parameters from "parameters"
        Wf = parameters["Wf"]
        bf = parameters["bf"]
        Wi = parameters["Wi"]
        bi = parameters["bi"]
        Wc = parameters["Wc"]
        bc = parameters["bc"]
        Wo = parameters["Wo"]
        bo = parameters["bo"]
        Wy = parameters["Wy"]
        by = parameters["by"]

        # Compute gates related derivatives, you can find their values can be found by looking carefully at equations (7) to (10) (≈4 lines)
        dot = da_next * np.tanh(c_next) * ot * (1 - ot)
        dcct = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * it * (1 - cct ** 2)
        dit = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * cct * (1 - it) * it
        dft = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * c_prev * ft * (1 - ft)

        # Code equations (7) to (10) (≈4 lines)
        dit = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * cct * (1 - it) * it
        dft = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * c_prev * ft * (1 - ft)
        dot = da_next * np.tanh(c_next) * ot * (1 - ot)
        dcct = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * it * (1 - cct ** 2)

        # Compute parameters related derivatives. Use equations (11)-(14) (≈8 lines)
        dWf = np.dot(dft, np.hstack([a_prev.T, xt.T]))
        dWi = np.dot(dit, np.hstack([a_prev.T, xt.T]))
        dWc = np.dot(dcct, np.hstack([a_prev.T, xt.T]))
        dWo = np.dot(dot, np.hstack([a_prev.T, xt.T]))
        dbf = np.sum(dft, axis=1, keepdims=True)
        dbi = np.sum(dit, axis=1, keepdims=True)
        dbc = np.sum(dcct, axis=1, keepdims=True)
        dbo = np.sum(dot, axis=1, keepdims=True)

        # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17). (≈3 lines)
        da_prev = np.dot(Wf[:, :n_a].T, dft) + np.dot(Wc[:, :n_a].T, dcct) + np.dot(Wi[:, :n_a].T, dit) + np.dot(
            Wo[:, :n_a].T, dot)
        dc_prev = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * ft
        dxt = np.dot(Wf[:, n_a:].T, dft) + np.dot(Wc[:, n_a:].T, dcct) + np.dot(Wi[:, n_a:].T, dit) + np.dot(
            Wo[:, n_a:].T, dot)
        ### END CODE HERE ###

        # Save gradients in dictionary
        gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                     "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

        return gradients



cfgConst = ConfigConstants()

MICROSECONDS_PER_MINUTE = cfgConst.getMicroSecPerMins()
time_per_time_slice = cfgConst.getTimePerTimeSlice()
highest_note = cfgConst.getHighestNote()
lowest_note = cfgConst.getLowestNote()  # A_2
input_dim = cfgConst.getInputDim() # number of notes in input
output_dim = cfgConst.getOutputDim()  # number of notes in output

x_seq_length = cfgConst.getXSeqLen()  # Piano roll matrix of dimention 50x49
y_seq_length = cfgConst.getYSeqLen()

num_units = 64

mi = MIDIDataProcess()

p = mi.getData('dataTest')
X, Y = mi.createXYDataset(p, x_seq_length, y_seq_length)

input_data = X.astype(np.bool)
target_data = Y.astype(np.bool)

print("input data shape : ", input_data.shape)
print("target data shape : ", target_data.shape)

print(input_data[:,1,:].shape)

lstm = LSTMUtil()

n_a = 100
params = lstm.initialize_parameters(n_a = n_a, n_x = input_data.shape[2], n_y = input_data.shape[2])

a0 = np.random.randn(input_data.shape[0], n_a )


a, y, c, caches = lstm.lstm_forward(X,a0, params)
