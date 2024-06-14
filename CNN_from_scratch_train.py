import numpy as np
import pandas as pd
import math
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from argparse import ArgumentParser

# Utility functions you will re-use
# Data-related utility functions
def load_blosum(filename):
    """
    Read in BLOSUM values into matrix.
    """
    aa = ['A', 'R', 'N' ,'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    df = pd.read_csv(filename, sep=r'\s+', comment='#', index_col=0)
    return df.loc[aa, aa]

def load_peptide_target(filename):
    """
    Read amino acid sequence of peptides and
    corresponding log transformed IC50 binding values from text file.
    """
    df = pd.read_csv(filename, sep=r'\s+', usecols=[0,1], names=['peptide','target'])
    return df.sort_values(by='target', ascending=False).reset_index(drop=True)

def encode_peptides(X_in, blosum_file, max_pep_len=9):
    """
    Encode AA seq of peptides using BLOSUM50.
    Returns a tensor of encoded peptides of shape (1, max_pep_len, n_features) for a single batch
    """
    blosum = load_blosum(blosum_file)
    
    batch_size = len(X_in)
    n_features = len(blosum)
    
    X_out = np.zeros((batch_size, max_pep_len, n_features), dtype=np.int8)
    
    for peptide_index, row in X_in.iterrows():
        for aa_index in range(len(row.peptide)):
            X_out[peptide_index, aa_index] = blosum.loc[row.peptide[aa_index]].values
            
    return X_out, np.expand_dims(X_in.target.values, 1)


# Misc. functions
def invoke(early_stopping, loss, model, implement=False):
    if not implement:
        return False
    else:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            return True
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_losses(train_losses, valid_losses, n_epochs):
    # Plotting the losses 
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(range(n_epochs), train_losses, label='Train loss', c='b')
    ax.plot(range(n_epochs), valid_losses, label='Valid loss', c='m')
    ax.legend()
    plt.show()


# Model saving and loading functions

def save_cnn_model(filepath, model):
    if not filepath.endswith('.pkl'):
        filepath = filepath + '.pkl'
    with open(filepath, 'wb') as f:
        dict_to_save = {
            'filter_size': model.filter.shape[0], 
            'input_size': model.filter.shape[1], 
            'n_filters': model.filter.shape[2], 
            'ffnn_hidden_size': model.ffnn.W1.shape[1], 
            'output_size': model.ffnn.W2.shape[1],
            'W1': model.ffnn.W1, 
            'b1': model.ffnn.b1, 
            'W2': model.ffnn.W2, 
            'b2': model.ffnn.b2, 
            "filter": model.filter,
        }
        pickle.dump(dict_to_save, f)
        print(f'Saved CNN model at {filepath}')


def load_cnn_model(filepath, model=None):

    with open(filepath, 'rb') as f:
        loaded_dict = pickle.load(f)
    if model is None:
        model = SimpleCNN_vectorized(
            loaded_dict['filter_size'], 
            loaded_dict['input_size'], 
            loaded_dict['n_filters'], 
            loaded_dict['ffnn_hidden_size'], 
            loaded_dict['output_size']
        )
    assert (
        model.filter.shape[0] == loaded_dict['filter_size'] and 
        model.filter.shape[1] == loaded_dict['input_size'] and 
        model.filter.shape[2] == loaded_dict['n_filters'] and 
        model.ffnn.W1.shape[1] == loaded_dict['ffnn_hidden_size'] and 
        model.ffnn.W2.shape[1] == loaded_dict['output_size']
    ), f"Model and loaded weights size mismatch!. Provided model has weight of dimensions {model.ffnn.W1.shape, model.ffnn.W2.shape, model.filter.shape} ; Loaded weights have shape {loaded_dict['W1'].shape, loaded_dict['W2'].shape, loaded_dict['filter'].shape}"

    model.ffnn.W1 = loaded_dict['W1']
    model.ffnn.b1 = loaded_dict['b1']
    model.ffnn.W2 = loaded_dict['W2']
    model.ffnn.b2 = loaded_dict['b2']
    model.filter = loaded_dict['filter']
    print(f"Model loaded successfully from {filepath}\nwith weights [ W1, W2, filter] dimensions : {model.ffnn.W1.shape, model.ffnn.W2.shape, model.filter.shape}")
    return model


# Weights initialization function.
# xavier initialization is technically more stable and preferred 
# (See slides)
def xavier_initialization_normal(filter_size, input_size, n_filters):
    shape = (filter_size, input_size, n_filters)
    fan_in = filter_size * input_size
    fan_out = filter_size * n_filters
    stddev = np.sqrt(2 / (fan_in + fan_out))
    return np.random.normal(0, stddev, size=shape) * 0.1

def xavier_initialization_normal_ffnn(input_dim, output_dim):
    shape = (input_dim, output_dim)
    stddev = np.sqrt(2 / (input_dim + output_dim))
    return np.random.normal(0, stddev, size=shape) * 0.1

def random_initialization_normal(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) * 0.1

def train_network(net, x_train, y_train, learning_rate):
    """
    Trains the network for a single epoch, running the forward and backward pass, and compute and return the loss.
    """
    # Forward pass
    conv_output, pooled_output, pool_indices, relu_output, z1, a1, z2, a2 = net.forward(x_train)
    # backward pass
    backward(net, x_train, y_train, conv_output, pooled_output, pool_indices, relu_output, z1, a1, z2, a2, learning_rate)
    loss = np.mean((a2 - y_train) ** 2)
    return loss
        
def eval_network(net, x_valid, y_valid):
    """
    Evaluates the network ; Note that we do not update weights (no backward pass)
    """
    conv_output, pooled_output, pool_indices, relu_output, z1, a1, z2, a2 = net.forward(x_valid)
    loss = np.mean((a2 - y_valid) ** 2)
    return loss

# FFN part
class SimpleFFNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with small random values
        self.W1 = xavier_initialization_normal_ffnn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = xavier_initialization_normal_ffnn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
    
    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x): 
        # This is equivalent to : 
        # if x>=0, then compute (1/(1+np.exp(-x)))
        # else: compute (np.exp(x)/(1+np.exp(x))))
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))

    def forward(self, x):
        """
        zi denotes the output of a hidden layer i
        ai denotes the output of an activation function at layer i
        (activations are relu, sigmoid, tanh, etc.)
        """

        # First layer
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self.relu(z1)
        
        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        
        # Return all the intermediate outputs as well because we need them for backpropagation (see slides)
        return z1, a1, z2, a2


class SimpleCNN:
    def __init__(self, filter_size, input_size, n_filters, hidden_size, output_size):
        self.filter = xavier_initialization_normal(filter_size, input_size, n_filters)
        self.bias = np.zeros((1, 1, n_filters))
        self.ffnn = SimpleFFNN(n_filters, hidden_size, output_size)

    def conv1d(self, batch_x):
        batch_size, input_length, input_size = batch_x.shape
        filter_size, _, n_filters = self.filter.shape
        output_length = input_length - filter_size + 1
        batch_output = np.zeros((batch_size, output_length, n_filters))

        # Loop through each sample in the batch
        for b in range(batch_size):
            for i in range(output_length):
                for k in range(n_filters):
                    # Extract the slice of input for the current position. Hint: we need to slice batch_x like [b, XX, :]
                    input_slice = batch_x[b, i:i + filter_size, :]
                    # Do element-wise multiplication of input slice filter k and sum. Hint: we need to slice the filter tensor and bias like this [:, :, k]
                    batch_output[b, i, k] = np.sum(input_slice * self.filter[:, :, k]) + self.bias[0, 0, k]

        return batch_output

    def relu(self, x):
        return np.maximum(0, x)

    def global_max_pooling(self, x):
        # Find the maximum value for each feature map and its index. Hint: look at np.max and np.argmax
        pool_indices = np.argmax(x, axis=1)
        max_pool = np.max(x, axis=1)
        return max_pool, pool_indices

    def forward(self, x):
        # Pass the input through the convolution layer
        conv_output = self.conv1d(x)
        # Apply global max pooling
        pooled_output, pool_indices = self.global_max_pooling(conv_output)
        # Apply the ReLU activation function
        relu_output = self.relu(pooled_output)
        
        # Pass the result through the feedforward neural network. Hint: unroll the output of the forward function like this *self.ffnn.forward(relu_output)
        return conv_output, pooled_output, pool_indices, relu_output, *self.ffnn.forward(relu_output)


def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid_derivative(x):
    return x * (1 - x)

def backward(net, x, y, conv_output, pooled_output, pool_indices, relu_output, z1, a1, z2, a2, learning_rate=0.01):
    # Backwards pass for the FFNN
    error = a2 - y
    d_output = error * sigmoid_derivative(a2)

    d_W2 = np.dot(a1.T, d_output)
    d_b2 = np.sum(d_output, axis=0, keepdims=True)

    error_hidden_layer = np.dot(d_output, net.ffnn.W2.T)
    d_hidden_layer = error_hidden_layer * relu_derivative(a1)

    d_W1 = np.dot(relu_output.T, d_hidden_layer)
    d_b1 = np.sum(d_hidden_layer, axis=0, keepdims=True)

    # Propagate the error back to the pooled layer. Hint: look at how we calculated the gradients for the previous layers.
    error_pool = np.dot(d_hidden_layer, net.ffnn.W1.T)
    d_pool = error_pool * relu_derivative(pooled_output)

    # Propagate the error to the convolution
    batch_size, input_length, input_size = x.shape
    filter_size, input_size, n_filters = net.filter.shape
    d_conv = np.zeros_like(conv_output)
    for b in range(batch_size):
        # Loop through each output channel
        for k in range(n_filters):
            # Get the index of the maximum value in the pooled output for this sample and channel. Hint: we kept track of this with pool_indices.
            max_index = pool_indices[b, k]
            # Fill out the d_conv matrix at the max indices using the d_pool value for input b and filter k
            d_conv[b, max_index, k] = d_pool[b, k]
    d_bias = np.sum(d_conv, axis=(0, 1), keepdims=True)

    # Calculate the gradient for the convolution filter. 
    d_filter = np.zeros_like(net.filter)
    output_length = input_length - filter_size + 1
    for b in range(batch_size):
        # Loop through each position in the convolution output
        for i in range(output_length):
            # Extract the input slice corresponding to the current position. Hint: we need to slice x like [b, i:i + filter_size, :]
            input_slice = x[b, i:i + filter_size, :]
            # Loop through each output channel
            for k in range(n_filters):
                # Accumulate the gradient for the filter by multiplying the input slice with the corresponding gradient from d_conv.
                # Hint for slicing d_conv: look at the dimensions of d_conv and which values are used in the loops.
                d_filter[:, :, k] += input_slice * d_conv[b, i, k]

    # Update the filter weights and the weights and biases of the FFNN
    net.filter -= learning_rate * d_filter
    net.bias -= learning_rate * d_bias.squeeze()
    net.ffnn.W1 -= learning_rate * d_W1
    net.ffnn.b1 -= learning_rate * d_b1.squeeze()
    net.ffnn.W2 -= learning_rate * d_W2
    net.ffnn.b2 -= learning_rate * d_b2.squeeze()


if __name__ == "__main__":
    parser = ArgumentParser(description="FFNN train script")
    # Data paths
    parser.add_argument("-train", action="store", dest="train_data", type=str, help="File with peptides (pep target)")
    parser.add_argument("-valid", action="store", dest="valid_data", type=str, help="File with peptides (pep target)")
    parser.add_argument("-savepath", action="store", dest="savepath", type=str, default='./CustomName', help='Path to save the result. Used to save the model as {savepath}_ffnn_model.pkl Must not have an extension, ex : ./path/to/my_file')

    # Model parameters
    parser.add_argument("-nh", action="store", dest="n_hidden", type=int, default=50, help="Number of hidden units")
    parser.add_argument("-nf", action="store", dest="n_filters", type=int, default=16, help="Number of CNN filters")
    parser.add_argument("-fs", action="store", dest="filter_size", type=int, default=5, help="CNN kernel size")
    parser.add_argument("-ne", action="store", dest="n_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("-lr", action="store", dest="learning_rate", type=float, default=0.0001, help="Learning rate")
    args = parser.parse_args()
    train_data = args.train_data
    valid_data = args.valid_data
    hidden_size = args.n_hidden
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    n_filters = args.n_filters
    filter_size = args.filter_size
    savepath = args.savepath

    # Load data
    DATAPATH = '/Users/anaselyoussef/Desktop/algo/data/NNDeep'  # Define DATAPATH here or modify accordingly
    blosum_file = f'{DATAPATH}/BLOSUM50'
    train_raw = load_peptide_target(train_data)
    valid_raw = load_peptide_target(valid_data)
    
    max_pep_len = train_raw.peptide.apply(len).max()
    x_train_, y_train_ = encode_peptides(train_raw, blosum_file, max_pep_len)
    x_valid_, y_valid_ = encode_peptides(valid_raw, blosum_file, max_pep_len)
    
    # Define sizes
    input_size = x_train_.shape[2] # also known as "n_features"
    
    # Creating a model instance 
    network = SimpleCNN(filter_size=filter_size, input_size=input_size, n_filters=n_filters, hidden_size=hidden_size, output_size=1)
    
    # Training loops
    train_losses = []
    valid_losses = []
    
    # Run n_epochs of training
    for epoch in range(n_epochs):
        train_loss = train_network(network, x_train_, y_train_, learning_rate)
        valid_loss = eval_network(network, x_valid_, y_valid_)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        # For the first, every 1% of the epochs and last epoch, we print the loss 
        # to check that the model is properly training. (loss going down)
        if (n_epochs >= 10 and epoch % math.ceil(0.01 * n_epochs) == 0) or epoch == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch}: \n\tTrain Loss:{train_loss:.4f}\tValid Loss:{valid_loss:.4f}")
    
    # saving the model to a file
    save_cnn_model(f'{savepath}_saved_cnn.pkl', model=network)
    
    # plotting the losses 
    plot_losses(train_losses, valid_losses, n_epochs)

# For Training
# python CNN_from_scratch_train.py -train /Users/anaselyoussef/Desktop/algo/data/NNDeep/A0301/train_BA -valid /Users/anaselyoussef/Desktop/algo/data/NNDeep/A0301/valid_BA -savepath /Users/anaselyoussef/Desktop/algo/outputfiles
