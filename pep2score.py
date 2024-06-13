import argparse
import numpy as np
from pprint import pprint
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process some files.")
parser.add_argument('-mat', dest='mat_file', required=True, help='File with PSSM')
parser.add_argument('-f', dest='peptides_file', required=True, help='File with peptides (format [peptide target])')
args = parser.parse_args()

def initialize_matrix(peptide_length, alphabet):
    init_matrix = [0] * peptide_length
    for i in range(peptide_length):
        row = {}
        for letter in alphabet: 
            row[letter] = 0.0
        init_matrix[i] = row
    return init_matrix

def from_psi_blast(file_name):
    with open(file_name, "r") as f:
        nline = 0
        for line in f:
            sline = str.split(line)
            if nline == 0:
                alphabet = [str] * len(sline)
                for i in range(len(sline)):
                    alphabet[i] = sline[i]
                matrix = initialize_matrix(peptide_length, alphabet)
            else:
                i = int(sline[0])
                for j in range(2, len(sline)):
                    matrix[i - 1][alphabet[j - 2]] = float(sline[j])
            nline += 1
    return matrix

def score_peptide(peptide, matrix):
    acum = 0
    for i in range(len(peptide)):
        acum += matrix[i][peptide[i]]
    return acum

# Read evaluation data
with open(args.peptides_file, 'r') as f:
    lines = f.readlines()

print("Lines read from the file:")
for line in lines:
    print(line.strip())

# Check and print the evaluation array
evaluation = np.loadtxt(args.peptides_file, dtype=str).reshape(-1, 2)
print("Evaluation array:")
print(evaluation)

evaluation_peptides = evaluation[:, 0]
evaluation_targets = evaluation[:, 1]

print("Evaluation peptides:")
print(evaluation_peptides)
print("Evaluation targets before conversion:")
print(evaluation_targets)

# Attempt to convert targets to float
try:
    evaluation_targets = evaluation_targets.astype(float)
except ValueError as e:
    print("Error converting targets to float:")
    print(e)
    exit(1)

peptide_length = len(evaluation_peptides[0])

# Define which PSSM file to use (file save from pep2mat)
w_matrix = from_psi_blast(args.mat_file)

evaluation_predictions = []
for i in range(len(evaluation_peptides)):
    score = score_peptide(evaluation_peptides[i], w_matrix)
    evaluation_predictions.append(score)
    print(evaluation_peptides[i], score, evaluation_targets[i])

pcc = pearsonr(evaluation_targets, evaluation_predictions)
print("PCC: ", pcc[0])

plt.scatter(evaluation_targets, evaluation_predictions)
plt.xlabel('Actual Targets')
plt.ylabel('Predicted Scores')
plt.title('Evaluation of Predicted Scores vs Actual Targets')
plt.show()
