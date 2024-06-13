import numpy as np
import pandas as pd
import pickle
import sys
from argparse import ArgumentParser

# Utility functions
def load_blosum(filename):
    aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    df = pd.read_csv(filename, sep=r'\s+', comment='#', index_col=0)
    return df.loc[aa, aa]

def load_peptide_target(filename):
    df = pd.read_csv(filename, sep=r'\s+', usecols=[0,1], names=['peptide','target'])
    return df.sort_values(by='target', ascending=False).reset_index(drop=True)

def validate_peptides(peptides):
    valid_aa = set("ARNDCEQGHILKMFPSTWYVX")
    valid_peptides = peptides['peptide'].apply(lambda seq: all(aa in valid_aa for aa in seq.upper()))
    return peptides[valid_peptides]

def encode_peptides(X_in, blosum_file, max_pep_len=9):
    blosum = load_blosum(blosum_file)
    batch_size = len(X_in)
    n_features = len(blosum)
    X_out = np.zeros((batch_size, max_pep_len, n_features), dtype=np.int8)
    for peptide_index, row in X_in.iterrows():
        for aa_index in range(len(row.peptide)):
            aa = row.peptide[aa_index].upper()  # Ensure amino acids are in uppercase
            if aa in blosum.columns:
                X_out[peptide_index, aa_index] = blosum[aa].values
            else:
                print(f"Warning: Amino acid '{aa}' not found in BLOSUM matrix. Skipping peptide '{row.peptide}'")
                break  # Skip encoding this peptide if it contains an invalid amino acid
    return X_out, np.expand_dims(X_in.target.values, 1)

class SimpleFFNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
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

def load_ffnn_model(filepath, model=None):
    with open(filepath, 'rb') as f:
        loaded_dict = pickle.load(f)
    if model is None:
        model = SimpleFFNN(loaded_dict['input_size'], loaded_dict['hidden_size'], loaded_dict['output_size'])
    model.W1 = loaded_dict['W1']
    model.b1 = loaded_dict['b1']
    model.W2 = loaded_dict['W2']
    model.b2 = loaded_dict['b2']
    return model

# Argument parsing
parser = ArgumentParser(description="FFNN inference script")
parser.add_argument("-data", action="store", dest="data_file", type=str, help="File with peptides for inference")
parser.add_argument("-params", action="store", dest="params_file", type=str, help="File with parameters for trained model")
parser.add_argument("-output", action="store", dest="output_dir", type=str, help="Directory to save the inference results")

args = parser.parse_args()

# Load data for inference
data_file = args.data_file
params_file = args.params_file
output_dir = args.output_dir

# Provide the correct path to your BLOSUM50 file
blosum_file = '/Users/anaselyoussef/Desktop/algo/data/NNDeep/BLOSUM50'

# Load inference data
inference_data = load_peptide_target(data_file)
# Validate peptides
inference_data = validate_peptides(inference_data)

if inference_data.empty:
    print("No valid peptides found. Exiting.")
    sys.exit()

max_pep_len = inference_data.peptide.apply(len).max()
x_infer, _ = encode_peptides(inference_data, blosum_file, max_pep_len)

# Check if valid peptides are found
if x_infer.size == 0:
    print("No valid peptides found. Exiting.")
    sys.exit()

# Reshape inference data
x_infer = x_infer.reshape(x_infer.shape[0], -1)

# Load the trained model
model = load_ffnn_model(params_file)

# Perform inference
_, _, _, predictions = model.forward(x_infer)

# Save predictions
inference_data['predictions'] = predictions
output_file = f"{output_dir}/inference_results.txt"
inference_data[['peptide', 'predictions']].to_csv(output_file, index=False, header=False)

print(f"Inference results saved to {output_file}")

# FOR TRAINING

# python FFNN_from_scratch.py -mode train -train /Users/anaselyoussef/Desktop/algo/data/ANN/A2403_training -valid /Users/anaselyoussef/Desktop/algo/data/ANN/A2403_evaluation -nh 16 -ne 500 -lr 0.0001 -savepath /Users/anaselyoussef/Desktop/algo/outputfiles/ANNtest1
