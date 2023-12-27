import numpy as np
from utils import *

def LSTM_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implementation of a single forward step of the LSTM-cell

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
    storage -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    
    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde), c stands for the cell state (memory)
    """

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

    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    concat = np.concatenate((a_prev, xt))

    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    c_next = np.add((ft * c_prev), (it * cct))
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = (ot* np.tanh(c_next))

    yt_pred = softmax(np.dot(Wy, a_next) + by)

    storage = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, storage

def LSTM_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (4).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
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
    c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
    store -- tuple of values needed for the backward pass, contains (list of all the storages, x)
    """

    store = []
    
    Wy = parameters['Wy']

    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    
    # initialization of "a", "c" and "y" with zeros
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    
    # Initialization of a_next and c_next
    a_next = a0
    c_next = np.zeros((n_a, m))
    
    # loop over all time-steps
    for t in range(T_x):

        a_next, c_next, yt, storage = LSTM_cell_forward(x[:,:,t], a_next, c_next, parameters)

        a[:,:,t] = a_next
        c[:,:,t]  = c_next
        y[:,:,t] = yt

        store.append(storage)

    store = (store, x)

    return a, y, c, store

def lstm_cell_backward(da_next, dc_next, storage):
    """
    Implement the backward pass for the LSTM-cell (single time-step).

    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    storage -- storage storing information from the forward pass

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

    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = storage
    
    concat_T = np.transpose(np.concatenate((a_prev, xt)))
    Wf = parameters["Wf"]
    Wc = parameters["Wc"]
    Wi = parameters["Wi"]
    Wo = parameters["Wo"]
    bc = parameters["bc"]
    bi = parameters["bi"]
    bf = parameters["bf"]
    bo = parameters["bo"]
    
    n_x, m = xt.shape
    n_a, m = a_next.shape
    
    # Computing gates related derivatives
    dit = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * cct * (1 - it) * it
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dft = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * c_prev * ft * (1 - ft)
    dcct = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * it * (1 - cct ** 2)
    
    # Computing parameters related derivatives
    dWf = np.dot(dft , concat_T)
    dWi = np.dot(dit , concat_T)
    dWc = np.dot(dcct , concat_T)
    dWo = np.dot(dot , concat_T)
    dbf = np.sum(dft, keepdims = True, axis = 1)
    dbi = np.sum(dit, keepdims = True, axis = 1)
    dbc = np.sum(dcct, keepdims = True, axis = 1)
    dbo = np.sum(dot, keepdims = True, axis = 1)

    # Computing derivatives w.r.t previous hidden state, previous memory state and inputs.
    da_prev = np.dot(Wf[:,:n_a].T, dft) + np.dot(Wi[:,:n_a].T, dit) + np.dot(Wc[:,:n_a].T, dcct) + np.dot(Wo[:,:n_a].T, dot)
    dc_prev = dc_next * ft + ot * np.subtract(1, np.square(np.tanh(c_next))) * ft * da_next
    dxt = np.dot(Wf[:,n_a::].T, dft) + np.dot(Wi[:,n_a::].T, dit) + np.dot(Wc[:,n_a::].T, dcct) + np.dot(Wo[:,n_a::].T, dot)

    
    # Save gradients in dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients

def lstm_backward(da, store):
    
    """
    Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).

    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    store -- storage storing information from the forward pass (lstm_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """
    
    (store, x) = store
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = store[0]
    
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))
    
    # loop back over the whole sequence
    for t in reversed(range(T_x)):
        # Computing all gradients using lstm_cell_backward.
        gradients = lstm_cell_backward(da[:,:,t] + da_prevt, dc_prevt, store[t])
        # Storing of the gradient to the parameters' previous step's gradient
        da_prevt = gradients["da_prev"]
        dc_prevt = gradients["dc_prev"]
        dx[:,:,t] = gradients["dxt"]
        dWf += gradients["dWf"]
        dWi += gradients["dWi"]
        dWc += gradients["dWc"]
        dWo += gradients["dWo"]
        dbf += gradients["dbf"]
        dbi += gradients["dbi"]
        dbc += gradients["dbc"]
        dbo += gradients["dbo"]

    # Setting the first activation's gradient to the backpropagated gradient da_prev.
    da0 = da_prevt

    # Storing the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
    
    return gradients