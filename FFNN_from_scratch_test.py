import numpy as np
import pandas as pd
import math
import pickle
import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Utility functions
def load_blosum(filename):
    aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    df = pd.read_csv(filename, sep=r'\s+', comment='#', index_col=0)
    return df.loc[aa, aa]

def load_peptide_target(filename):
    df = pd.read_csv(filename, sep=r'\s+', usecols=[0,1], names=['peptide','target'])
    return df.sort_values(by='target', ascending=False).reset_index(drop=True)

def encode_peptides(X_in, blosum_file, max_pep_len=9):
    blosum = load_blosum(blosum_file)
    batch_size = len(X_in)
    n_features = len(blosum)
    X_out = np.zeros((batch_size, max_pep_len, n_features), dtype=np.int8)
    for peptide_index, row in X_in.iterrows():
        for aa_index in range(len(row.peptide)):
            aa = row.peptide[aa_index].upper()
            if aa in blosum.columns:
                X_out[peptide_index, aa_index] = blosum[aa].values
            else:
                print(f"Warning: Amino acid '{aa}' not found in BLOSUM matrix. Skipping peptide '{row.peptide}'")
                break
    return X_out, np.expand_dims(X_in.target.values, 1)

# Define the SimpleFFNN class and other functions as before...

def xavier_initialization_normal(input_dim, output_dim):
    shape = (input_dim, output_dim)
    stddev = np.sqrt(2 / (input_dim + output_dim))
    return np.random.normal(0, stddev, size=shape) * 0.1

class SimpleFFNN:
    def __init__(self, input_size, hidden_size, output_size, initialization_function=xavier_initialization_normal):
        self.W1 = initialization_function(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = initialization_function(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
        
    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def forward(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        return z1, a1, z2, a2

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid_derivative(x):
    sig = 1 / (1 + np.exp(-x))
    return sig * (1 - sig)

def backward(net, x, y, z1, a1, z2, a2, learning_rate=0.01):
    error = a2 - y
    d_output = error * sigmoid_derivative(z2) 
    d_W2 = np.dot(a1.T, d_output)
    d_b2 = np.sum(d_output, axis=0, keepdims=True)
    d_b2 = d_b2.squeeze()
    error_hidden_layer = np.dot(d_output, net.W2.T)
    d_hidden_layer = error_hidden_layer * relu_derivative(z1)
    d_W1 = np.dot(x.T, d_hidden_layer)
    d_b1 = np.sum(d_hidden_layer, axis=0, keepdims=True)
    d_b1 = d_b1.squeeze()
    net.W1 -= learning_rate * d_W1
    net.b1 -= learning_rate * d_b1
    net.W2 -= learning_rate * d_W2
    net.b2 -= learning_rate * d_b2

def train_network(net, x_train, y_train, learning_rate):
    z1, a1, z2, a2 = net.forward(x_train)
    backward(net, x_train, y_train, z1, a1, z2, a2, learning_rate)
    loss = np.mean((a2 - y_train) ** 2)
    return loss

def eval_network(net, x_valid, y_valid):
    z1, a1, z2, a2 = net.forward(x_valid)
    loss = np.mean((a2 - y_valid) ** 2)
    return loss

def save_ffnn_model(filepath, model):
    if not filepath.endswith('.pkl'):
        filepath = filepath + '.pkl'
    with open(filepath, 'wb') as f:
        dict_to_save = {'input_size': model.W1.shape[0], 'hidden_size': model.W1.shape[1], 'output_size': model.W2.shape[1],
                        'W1': model.W1, 'b1': model.b1, 'W2': model.W2, 'b2': model.b2}
        pickle.dump(dict_to_save, f)
        print(f'Saved FFNN model at {filepath}')

def plot_losses(train_losses, valid_losses, n_epochs):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(range(n_epochs), train_losses, label='Train loss', c='b')
    ax.plot(range(n_epochs), valid_losses, label='Valid loss', c='m')
    ax.legend()
    plt.show()

# Argument parsing
parser = ArgumentParser(description="FFNN train script")
parser.add_argument("-train", action="store", dest="train_data", type=str, help="File with peptides (pep target)")
parser.add_argument("-valid", action="store", dest="valid_data", type=str, help="File with peptides (pep target)")
parser.add_argument("-nh", action="store", dest="n_hidden", type=int, default=16, help="Number of hidden units")
parser.add_argument("-ne", action="store", dest="n_epochs", type=int, default=500, help="Number of epochs")
parser.add_argument("-lr", action="store", dest="learning_rate", type=float, default=0.0001, help="Learning rate")
parser.add_argument("-savepath", action="store", dest="savepath", type=str, default='./CustomName', help='Path to save the result. Used to save the model as {savepath}_ffnn_model.pkl Must not have an extension, ex : ./path/to/my_file')
args = parser.parse_args()

# Replace your data paths with the actual paths and desired alleles
train_data = args.train_data
valid_data = args.valid_data
hidden_size = args.n_hidden
n_epochs = args.n_epochs
learning_rate = args.learning_rate
savepath = args.savepath

# Provide the correct path to your BLOSUM50 file
blosum_file = '/Users/anaselyoussef/Desktop/algo/data/NNDeep/BLOSUM50'

# Loading the peptides.
train_raw = load_peptide_target(train_data)
valid_raw = load_peptide_target(valid_data)

print('Preview of the dataframe ; Peptides have to be *encoded* to BLOSUM matrices')
print(train_raw.head())

print('N_datapoints:')
print('Train data:\t', train_raw.shape[0])
print('Valid data:\t', valid_raw.shape[0])

print('Maximum peptide length of each data set:')
print('Train:\t',  train_raw['peptide'].apply(len).max())
print('Valid:\t', valid_raw['peptide'].apply(len).max())

train_raw['len'] = train_raw['peptide'].apply(len)
print('Peptide length counts in the train data')
print(train_raw.groupby('len').agg(count=('peptide', 'count')))

max_pep_len = train_raw.peptide.apply(len).max()
x_train_, y_train_ = encode_peptides(train_raw, blosum_file, max_pep_len)
x_valid_, y_valid_ = encode_peptides(valid_raw, blosum_file, max_pep_len)

# Reshaping the matrices so they're flat because feed-forward networks are "one-dimensional"
x_train_ = x_train_.reshape(x_train_.shape[0], -1)
x_valid_ = x_valid_.reshape(x_valid_.shape[0], -1)
# Define sizes
input_size = x_train_.shape[1] # also known as "n_features"
# Model and training hyperparameters
output_size = 1

# Creating a model instance 
network = SimpleFFNN(input_size, hidden_size, output_size)

# Training loops
train_losses = []
valid_losses = []

# Run n_epochs of training
for epoch in range(n_epochs):
    train_loss = train_network(network, x_train_, y_train_, learning_rate)
    valid_loss = eval_network(network, x_valid_, y_valid_)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    if (n_epochs >= 10 and epoch % math.ceil(0.05 * n_epochs) == 0) or epoch == 0 or epoch == n_epochs:
        print(f"Epoch {epoch}: \n\tTrain Loss:{train_loss:.4f}\tValid Loss:{valid_loss:.4f}")

# saving the model to a file
save_ffnn_model(f'{savepath}_ffnn_model.pkl', model=network)

# plotting the losses 
plot_losses(train_losses, valid_losses, n_epochs)


# FOR TRAINING
# python FFNN_from_scratch_test.py -mode train -train /Users/anaselyoussef/Desktop/algo/data/ANN/A2403_training -valid /Users/anaselyoussef/Desktop/algo/data/ANN/A2403_evaluation -nh 16 -ne 500 -lr 0.0001 -savepath /Users/anaselyoussef/Desktop/algo/outputfiles/ANNtest22


