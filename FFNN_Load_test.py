import numpy as np
import pandas as pd
import pickle
import sys
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Utility functions
def load_blosum(filename):
    aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    df = pd.read_csv(filename, sep=r'\s+', comment='#', index_col=0)
    return df.loc[aa, aa]

def load_peptide_target(filename):
    df = pd.read_csv(filename, sep=r'\s+', usecols=[0, 1, 2], names=['peptide', 'target', 'allele'])
    df = df.drop(columns=['allele'])
    return df.sort_values(by='target', ascending=False).reset_index(drop=True)

def encode_peptides(X_in, blosum_file, max_pep_len=9):
    blosum = load_blosum(blosum_file)
    batch_size = len(X_in)
    n_features = len(blosum)
    X_out = np.zeros((batch_size, max_pep_len, n_features), dtype=np.int8)
    for peptide_index, row in X_in.iterrows():
        for aa_index in range(len(row.peptide)):
            X_out[peptide_index, aa_index] = blosum[row.peptide[aa_index]].values
    return X_out, np.expand_dims(X_in.target.values, 1)

def load_ffnn_model(filepath, model=None):
    with open(filepath, 'rb') as f:
        loaded_dict = pickle.load(f)
    if model is None:
        model = SimpleFFNN(loaded_dict['input_size'], loaded_dict['hidden_size'], loaded_dict['output_size'])
    model.W1 = loaded_dict['W1']
    model.b1 = loaded_dict['b1']
    model.W2 = loaded_dict['W2']
    model.b2 = loaded_dict['b2']
    print(f"Model loaded successfully from {filepath}\nwith weights [ W1, W2 ] dimensions : {model.W1.shape, model.W2.shape}")
    return model

class SimpleFFNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.zeros((input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.zeros((hidden_size, output_size))
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

# Argument parsing
parser = ArgumentParser(description="FFNN test script")
parser.add_argument("-data", action="store", dest="data_file", type=str, help="File with test peptides (pep target)")
parser.add_argument("-params", action="store", dest="params_file", type=str, help="File with trained model parameters")
parser.add_argument("-output", action="store", dest="output_dir", type=str, help="Path to output directory")
args = parser.parse_args()

data_file = args.data_file
params_file = args.params_file
output_dir = args.output_dir

# Provide the correct path to your BLOSUM50 file
blosum_file = '/Users/anaselyoussef/Desktop/algo/data/NNDeep/BLOSUM50'

# Loading the peptides
test_raw = load_peptide_target(data_file)
if test_raw.empty:
    print("No valid peptides found. Exiting.")
    sys.exit(1)

print('N_datapoints:')
print('Test data:\t', test_raw.shape[0])

print('Maximum peptide length of the test data set:')
print('Test:\t', test_raw['peptide'].apply(len).max())

max_pep_len = test_raw.peptide.apply(len).max()
x_test_, y_test_ = encode_peptides(test_raw, blosum_file, max_pep_len)
x_test_ = x_test_.reshape(x_test_.shape[0], -1)

# Load the trained model
reloaded_network = load_ffnn_model(params_file)

# Perform inference
_, _, _, test_predictions_scores = reloaded_network.forward(x_test_)

# Threshold the target values to create binary labels
BINDER_THRESHOLD = 0.426
y_test_binary = (y_test_ >= BINDER_THRESHOLD).astype(int)

# Calculate ROC AUC
test_auc = roc_auc_score(y_test_binary.squeeze(), test_predictions_scores.squeeze())
test_fpr, test_tpr, _ = roc_curve(y_test_binary.squeeze(), test_predictions_scores.squeeze())

f, a = plt.subplots(1, 1, figsize=(9, 9))
a.plot([0, 1], [0, 1], ls=':', lw=0.5, label='Random prediction: AUC=0.500', c='k')
a.plot(test_fpr, test_tpr, ls='--', lw=1, label=f'Neural Network: AUC={test_auc:.3f}', c='b')
a.legend()
plt.show()

# Save the predictions
test_raw['predictions'] = test_predictions_scores
output_file = f'{output_dir}/test_predictions.txt'
test_raw[['peptide', 'predictions', 'target']].to_csv(output_file, index=False, header=False)
print(f"Predictions saved to {output_file}")

# For Testing

# python python FFNN_Load_test.py -data /Users/anaselyoussef/Desktop/algo/data/NNDeep/A0301/test_BA -params /Users/anaselyoussef/Desktop/algo/outputfiles/ANNtest1_ffnn_model.pkl -output /Users/anaselyoussef/Desktop/algo/outputfiles