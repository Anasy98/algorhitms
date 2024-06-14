import numpy as np
import pandas as pd
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, roc_auc_score
from argparse import ArgumentParser

# Utility functions you will re-use
def load_blosum(filename):
    aa = ['A', 'R', 'N' ,'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
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
            X_out[peptide_index, aa_index] = blosum.loc[row.peptide[aa_index]].values
    return X_out, np.expand_dims(X_in.target.values, 1)

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

class SimpleFFNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = xavier_initialization_normal_ffnn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = xavier_initialization_normal_ffnn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
    
    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x): 
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))

    def forward(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        return z1, a1, z2, a2

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

class SimpleCNN_vectorized:
    def __init__(self, filter_size, input_size, n_filters, hidden_size, output_size):
        self.filter = xavier_initialization_normal(filter_size, input_size, n_filters)
        self.bias = np.zeros((1, 1, n_filters))
        self.ffnn = SimpleFFNN(n_filters, hidden_size, output_size)

    def conv1d(self, batch_x):
        batch_size, input_length, input_size = batch_x.shape
        filter_size, _, n_filters = self.filter.shape
        output_length = input_length - filter_size + 1
        batch_output = np.zeros((batch_size, output_length, n_filters))

        for i in range(output_length):
            input_slice = batch_x[:, i:i + filter_size, :, None]
            batch_output[:, i, :] = np.sum(input_slice * self.filter, axis=(1, 2)) + self.bias
    
        return batch_output

    def relu(self, x):
        return np.maximum(0, x)

    def global_max_pooling(self, x):
        pool_indices = np.argmax(x, axis=1)
        max_pool = np.max(x, axis=1)
        return max_pool, pool_indices

    def forward(self, x):
        conv_output = self.conv1d(x)
        pooled_output, pool_indices = self.global_max_pooling(conv_output)
        relu_output = self.relu(pooled_output)
        return conv_output, pooled_output, pool_indices, relu_output, *self.ffnn.forward(relu_output)

if __name__ == "__main__":
    parser = ArgumentParser(description="FFNN test script")
    parser.add_argument("-test", action="store", dest="test_data", type=str, help="File with peptides (pep target)")
    parser.add_argument("-savepath", action="store", dest="savepath", type=str, default='./CustomName', help='Path to save the result. Used to load the model as {savepath}_saved_cnn.pkl and save the predictions as {savepath}_ffnn_predictions.txt ; Must not have an extension, ex : ./path/to/my_file')
    args = parser.parse_args()
    test_data = args.test_data
    savepath = args.savepath

    # Load data
    DATAPATH = '/Users/anaselyoussef/Desktop/algo/data/NNDeep'
    blosum_file = f'{DATAPATH}/BLOSUM50'
    test_raw = load_peptide_target(test_data)
    
    max_pep_len = test_raw.peptide.apply(len).max()
    x_test_, y_test_ = encode_peptides(test_raw, blosum_file, max_pep_len)

    # Reload the model and evaluate it
    reloaded_network = load_cnn_model(f'{savepath}_saved_cnn.pkl')

    # Thresholding the targets
    BINDER_THRESHOLD = 0.426
    y_test_thresholded = (y_test_ >= BINDER_THRESHOLD).astype(int)
    _, _, _, _, _, _, _, test_predictions = reloaded_network.forward(x_test_)

    # Saving the predictions
    test_raw['predictions'] = test_predictions
    test_raw[['peptide', 'predictions', 'target']].to_csv(f'{savepath}_ffnn_predictions.txt', index=False, header=False)

    # Plot ROC curve
    test_auc = roc_auc_score(y_test_thresholded.squeeze(), test_predictions.squeeze())
    test_fpr, test_tpr, _ = roc_curve(y_test_thresholded.squeeze(), test_predictions.squeeze())

    f, a = plt.subplots(1, 1, figsize=(9, 9))
    a.plot([0, 1], [0, 1], ls=':', lw=0.5, label='Random prediction: AUC=0.500', c='k')
    a.plot(test_fpr, test_tpr, ls='--', lw=1, label=f'Neural Network: AUC={test_auc:.3f}', c='b')
    a.legend()
    plt.show()

# For Testing

# python CNN_from_scratch_test.py -test /Users/anaselyoussef/Desktop/algo/data/NNDeep/A0301/test_BA -savepath /Users/anaselyoussef/Desktop/algo/outputfiles