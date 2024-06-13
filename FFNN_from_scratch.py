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
            X_out[peptide_index, aa_index] = blosum[ row.peptide[aa_index] ].values
            
    return X_out, np.expand_dims(X_in.target.values,1)

# Misc. functions
def invoke(early_stopping, loss, model, implement=False):
    if implement == False:
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
    fig,ax = plt.subplots(1,1, figsize=(9,5))
    ax.plot(range(n_epochs), train_losses, label='Train loss', c='b')
    ax.plot(range(n_epochs), valid_losses, label='Valid loss', c='m')
    ax.legend()
    plt.show()

# Weights initialization function.
# xavier initialization is technically more stable and preferred 
# (See slides)
def xavier_initialization_normal(input_dim, output_dim):
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
    z1, a1, z2, a2 = net.forward(x_train)
    # backward pass
    backward(net, x_train, y_train, z1, a1, z2, a2, learning_rate)
    loss = np.mean((a2 - y_train) ** 2)
    return loss
        
def eval_network(net, x_valid, y_valid):
    """
    Evaluates the network ; Note that we do not update weights (no backward pass)
    """
    z1, a1, z2, a2 = net.forward(x_valid)
    loss = np.mean((a2-y_valid)**2)
    return loss

# Model saving and loading functions

def save_ffnn_model(filepath, model):
    if not filepath.endswith('.pkl'):
        filepath = filepath+'.pkl'
    with open(filepath, 'wb') as f:
        dict_to_save = {'input_size': model.W1.shape[0], 'hidden_size':model.W1.shape[1], 'output_size':model.W2.shape[1],
                        'W1': model.W1, 'b1':model.b1, 'W2':model.W2, 'b2':model.b2}
        pickle.dump(dict_to_save, f)
        print(f'Saved FFNN model at {filepath}')


def load_ffnn_model(filepath, model=None):

    with open(filepath, 'rb') as f:
        loaded_dict = pickle.load(f)
    if model is None:
            model = SimpleFFNN(loaded_dict['input_size'], loaded_dict['hidden_size'], loaded_dict['output_size'])
    assert (model.W1.shape[0]==loaded_dict['input_size'] and model.W1.shape[1]==loaded_dict['hidden_size'] and model.W2.shape[1]==loaded_dict['output_size']), \
        f"Model and loaded weights size mismatch!. Provided model has weight of dimensions {model.W1.shape, model.W2.shape} ; Loaded weights have shape {loaded_dict['W1'].shape, loaded_dict['W2'].shape}"

    model.W1 = loaded_dict['W1']
    model.b1 = loaded_dict['b1']
    model.W2 = loaded_dict['W2']
    model.b2 = loaded_dict['b2']
    print(f"Model loaded successfully from {filepath}\nwith weights [ W1, W2 ] dimensions : {model.W1.shape, model.W2.shape}")
    return model

# Argument parsing
parser = ArgumentParser(description="FFNN train script")
parser.add_argument("-train", action="store", dest="train_data", type=str, help="File with peptides (pep target)")
parser.add_argument("-valid", action="store", dest="valid_data", type=str, help="File with peptides (pep target)")
parser.add_argument("-test", action="store", dest="test_data", type=str, help="File with peptides (pep target)")
parser.add_argument("-nh", action="store", dest="n_hidden", type=int, default=16, help="Number of hidden units")
parser.add_argument("-ne", action="store", dest="n_epochs", type=int, default=500, help="Number of epochs")
parser.add_argument("-lr", action="store", dest="learning_rate", type=float, default=0.0001, help="Learning rate")
parser.add_argument("-savepath", action="store", dest="savepath", type=str, default='./CustomName', help='Path to save the result. Used to save the model as {savepath}_ffnn_model.pkl Must not have an extension, ex : ./path/to/my_file')
args = parser.parse_args()

# Replace your data paths with the actual paths and desired alleles
train_data = args.train_data
valid_data = args.valid_data
test_data = args.test_data
hidden_size = args.n_hidden
n_epochs = args.n_epochs
learning_rate = args.learning_rate
savepath = args.savepath

# Provide the correct path to your BLOSUM50 file
blosum_file = '/Users/anaselyoussef/Desktop/algo/data/NNDeep/BLOSUM50'

# Loading the peptides.
train_raw = load_peptide_target(train_data)
valid_raw = load_peptide_target(valid_data)
test_raw = load_peptide_target(test_data)

print('Preview of the dataframe ; Peptides have to be *encoded* to BLOSUM matrices')
print(train_raw.head())

print('N_datapoints:')
print('Train data:\t', train_raw.shape[0])
print('Valid data:\t', valid_raw.shape[0])
print('Test data:\t', test_raw.shape[0])

print('Maximum peptide length of each data set:')
print('Train:\t',  train_raw['peptide'].apply(len).max())
print('Valid:\t', valid_raw['peptide'].apply(len).max())
print('Test:\t', test_raw['peptide'].apply(len).max())

train_raw['len'] = train_raw['peptide'].apply(len)
print('Peptide length counts in the train data')
print(train_raw.groupby('len').agg(count=('peptide', 'count')))

max_pep_len = train_raw.peptide.apply(len).max()
x_train_, y_train_ = encode_peptides(train_raw, blosum_file, max_pep_len)
x_valid_, y_valid_ = encode_peptides(valid_raw, blosum_file, max_pep_len)
x_test_, y_test_ = encode_peptides(test_raw, blosum_file, max_pep_len)
# We now have matrices of shape (N_datapoints, max_pep_len, n_features)
print(x_train_.shape)

class SimpleFFNN:
    def __init__(self, input_size, hidden_size, output_size, initialization_function=xavier_initialization_normal):
        # Initialize weights and biases with small random values
        # initialization_function(input_dim, output_dim) -> np.array of shape (input_dim, output_dim)
        self.W1 = initialization_function(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = initialization_function(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
        print(self.W1.shape, self.b1.shape, self.W2.shape, self.b2.shape)
        
    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x): 
        """
        The normal version of sigmoid 1 / (1 + np.exp(-x)) is NOT numerically stable
        Here we split the case into two for positive and negative inputs
        because np.exp(-x) for something negative will quickly overflow if x is a large negative number
        """
        # This is equivalent to : 
        # if x>=0, then compute (1/(1+np.exp(-x)))
        # if x<0: compute (np.exp(x)/(1+np.exp(x))))
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))

    def forward(self, x):
        """
        zi denotes the output of a hidden layer i
        ai denotes the output of an activation function (non-linearity) at layer i
        (activations are relu, sigmoid, tanh, etc.)
        Use self.function to call a method. for example: self.relu(XX)
        """

        # First layer : Use a relu here for the activation
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self.relu(z1)
        
        # Output layer : Use a sigmoid here for the activation
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        
        # Return all the intermediate outputs as well because we need them for backpropagation (see slides)
        return z1, a1, z2, a2

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid_derivative(x):
    """
    For this derivative, it is not necessary to find a numerically stable version.
    Just take the base formula and derive it.
    """
    sig = 1 / (1 + np.exp(-x))
    return sig * (1 - sig)

def backward(net, x, y, z1, a1, z2, a2, learning_rate=0.01):
    """
    Function to backpropagate the gradients from the output to update the weights.
    Apply the chain rule and slowly work out the chain derivatives from the output back to the input
    Reminder that np.dot(array_1, array_2) and array.T exists to transpose an array for matrix multiplication
    """
    # This assumes that we are computing a MSE as the loss function.
    # Look at your slides to compute the gradient backpropagation for a mean-squared error using the chain rule.

    # Output layer error ; We used a sigmoid in this layer
    error = a2 - y
    d_output = error * sigmoid_derivative(z2) 

    # Backpropagate to hidden layer
    d_W2 = np.dot(a1.T, d_output)
    d_b2 = np.sum(d_output, axis=0, keepdims=True)
    d_b2 = d_b2.squeeze() # Squeeze is needed here to make the dimensions fit

    # Hidden layer error ; We used a ReLU in this layer!
    error_hidden_layer = np.dot(d_output, net.W2.T)
    d_hidden_layer = error_hidden_layer * relu_derivative(z1)

    # Backpropagate to input layer
    d_W1 = np.dot(x.T, d_hidden_layer)
    d_b1 = np.sum(d_hidden_layer, axis=0, keepdims=True)
    d_b1 = d_b1.squeeze() # Squeeze is needed here to make the dimensions fit
    
    # Update weights and biases using gradient descent
    net.W1 -= learning_rate * d_W1
    net.b1 -= learning_rate * d_b1
    net.W2 -= learning_rate * d_W2
    net.b2 -= learning_rate * d_b2


# Reshaping the matrices so they're flat because feed-forward networks are "one-dimensional"
x_train_ = x_train_.reshape(x_train_.shape[0], -1)
x_valid_ = x_valid_.reshape(x_valid_.shape[0], -1)
x_test_ = x_test_.reshape(x_test_.shape[0], -1)
# Define sizes
input_size = x_train_.shape[1] # also known as "n_features"
# Model and training hyperparameters
learning_rate = learning_rate
hidden_units = hidden_size
n_epochs = n_epochs
output_size = 1
# Creating a model instance 
# You can use either `xavier_initialization_normal` or `random_initialization_normal`
# for the initialization_function argument of the class
network = SimpleFFNN(input_size, hidden_units, output_size)


# Training loops
train_losses = []
valid_losses = []

# Run n_epochs of training
for epoch in range(n_epochs):
    train_loss = train_network(network, x_train_, y_train_, learning_rate)
    valid_loss = eval_network(network, x_valid_, y_valid_)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    # For the first, every 5% of the epochs and last epoch, we print the loss 
    # to check that the model is properly training. (loss going down)
    if (n_epochs >= 10 and epoch % math.ceil(0.05 * n_epochs) == 0) or epoch == 0 or epoch == n_epochs:
        print(f"Epoch {epoch}: \n\tTrain Loss:{train_loss:.4f}\tValid Loss:{valid_loss:.4f}")

# saving the model to a file
# Use the .pkl extension to save python pickled files
save_ffnn_model(f'{savepath}_ffnn_model.pkl', model=network)


# plotting the losses 
plot_losses(train_losses, valid_losses, n_epochs)


from sklearn.metrics import roc_auc_score, roc_curve
# Reload the model and evaluate it
reloaded_network = load_ffnn_model(f'{savepath}_ffnn_model.pkl', model=network)


BINDER_THRESHOLD=0.426
# Thresholding the targets
y_test_thresholded = (y_test_>=BINDER_THRESHOLD).astype(int)
_, _, _, test_predictions_scores = reloaded_network.forward(x_test_)
test_auc = roc_auc_score(y_test_thresholded.squeeze(), test_predictions_scores.squeeze())
test_fpr, test_tpr, _ = roc_curve(y_test_thresholded.squeeze(), test_predictions_scores.squeeze())

f,a = plt.subplots(1,1 , figsize=(9,9))

a.plot([0,1],[0,1], ls=':', lw=0.5, label='Random prediction: AUC=0.500', c='k')
a.plot(test_fpr, test_tpr, ls='--', lw=1, label=f'Neural Network: AUC={test_auc:.3f}', c='b')
a.legend()

# Saving the predictions
test_raw['predictions'] = test_predictions_scores
test_raw[['peptide','predictions','target']].to_csv(f'{savepath}_ffnn_predictions.txt', index=False, header=False)

# Training loops
results={}
learning_rate=0.0001
n_epochs = 250


BINDER_THRESHOLD=0.426
# Thresholding the targets
y_test_thresholded = (y_test_>=BINDER_THRESHOLD).astype(int)

f,a = plt.subplots(1,1, figsize=(9,9))
a.plot([0,1],[0,1],ls=':', c='k', lw=0.5, label='Random prediction: AUC=0.500')

# Tuning hyperparameters by looking at which one performs best
for learning_rate in [0.01, 0.0001]:
    for hidden_size in [10, 20, 64]:
        print(f'Training model with hidden size : {hidden_size}, learning_rate : {learning_rate}')
        model = SimpleFFNN(input_size, hidden_size, output_size)
        for _ in range(n_epochs):
            train_network(model, x_train_, y_train_, learning_rate)
        _, _, _, test_predictions_scores = model.forward(x_test_)
        test_auc = roc_auc_score(y_test_thresholded.squeeze(), test_predictions_scores.squeeze())
        test_fpr, test_tpr, _ = roc_curve(y_test_thresholded.squeeze(), test_predictions_scores.squeeze())
        a.plot(test_fpr, test_tpr, ls='--', lw=1, label=f'Model {hidden_size} ; lr={learning_rate}: AUC={test_auc:.3f}')
a.legend()
plt.show()


# For Training 
# python FFNN_from_scratch.py -train /Users/anaselyoussef/Desktop/algo/data/NNDeep/A0301/train_BA -valid /Users/anaselyoussef/Desktop/algo/data/NNDeep/A0301/valid_BA -test /Users/anaselyoussef/Desktop/algo/data/NNDeep/A0301/test_BA -nh 16 -ne 500 -lr 0.0001 -savepath /Users/anaselyoussef/Desktop/algo/outputfiles/ANNtest5
