import numpy as np
import random
import copy
from scipy.stats import pearsonr
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SMM_GD")
    parser.add_argument('-l', type=float, default=0.01, help='Lambda (default: 0.01)')
    parser.add_argument('-t', required=True, help='File with training data')
    parser.add_argument('-e', required=True, help='File with evaluation data')
    parser.add_argument('-epi', type=float, default=0.05, help='Epsilon (default 0.05)')
    parser.add_argument('-s', type=int, default=1, help='Seed for random numbers (default 1)')
    parser.add_argument('-i', type=int, default=100, help='Number of epochs to train (default 100)')
    return parser.parse_args()

def encode(peptides, encoding_scheme, alphabet):
    encoded_peptides = []
    for peptide in peptides:
        encoded_peptide = []
        for peptide_letter in peptide:
            for alphabet_letter in alphabet:
                encoded_peptide.append(encoding_scheme[peptide_letter][alphabet_letter])
        encoded_peptides.append(encoded_peptide)
    return np.array(encoded_peptides)

def cumulative_error(peptides, y, lamb, weights):
    error = 0
    for i in range(0, len(peptides)):
        peptide = peptides[i]
        y_target = y[i]
        y_pred = np.dot(peptide, weights)
        error += 1.0/2 * (y_pred - y_target)**2
    gerror = error + lamb*np.dot(weights, weights)
    error /= len(peptides)
    return gerror, error

def predict(peptides, weights):
    pred = []
    for i in range(0, len(peptides)):
        peptide = peptides[i]
        y_pred = np.dot(peptide, weights)
        pred.append(y_pred)
    return pred

def cal_mse(vec1, vec2):
    mse = 0
    for i in range(0, len(vec1)):
        mse += (vec1[i] - vec2[i])**2
    mse /= len(vec1)
    return mse

def gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon):
    do = y_pred - y_target
    for i in range(0, len(weights)):
        de_dw_i = do*peptide[i] + (2*lamb_N)*weights[i]
        weights[i] -= epsilon * de_dw_i

def vector_to_matrix(vector, alphabet):
    rows = int(len(vector) / len(alphabet))
    matrix = [0] * rows
    offset = 0
    for i in range(0, rows):
        matrix[i] = {}
        for j in range(0, 20):
            matrix[i][alphabet[j]] = vector[j + offset]
        offset += len(alphabet)
    return matrix

def to_psi_blast(matrix):
    header = ["", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*header))
    letter_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    for i, row in enumerate(matrix):
        scores = []
        scores.append(str(i + 1) + " A")
        for letter in letter_order:
            score = row[letter]
            scores.append(round(score, 4))
        print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*scores))

def main():
    args = parse_args()

    data_dir = "/Users/anaselyoussef/Desktop/algo/data/"
    training_file = args.t
    evaluation_file = args.e

    training = np.loadtxt(training_file, dtype=str)
    evaluation = np.loadtxt(evaluation_file, dtype=str)

    alphabet_file = data_dir + "Matrices/alphabet"
    alphabet = np.loadtxt(alphabet_file, dtype=str)

    sparse_file = data_dir + "Matrices/sparse"
    _sparse = np.loadtxt(sparse_file, dtype=float)
    sparse = {}

    for i, letter_1 in enumerate(alphabet):
        sparse[letter_1] = {}
        for j, letter_2 in enumerate(alphabet):
            sparse[letter_1][letter_2] = _sparse[i, j]

    np.random.seed(args.s)
    peptides = training[:, 0]
    peptides = encode(peptides, sparse, alphabet)
    N = len(peptides)
    y = np.array(training[:, 1], dtype=float)
    evaluation_peptides = evaluation[:, 0]
    evaluation_peptides = encode(evaluation_peptides, sparse, alphabet)
    evaluation_targets = np.array(evaluation[:, 1], dtype=float)
    input_dim = len(peptides[0])
    output_dim = 1
    w_bound = 0.1
    weights = np.random.uniform(-w_bound, w_bound, size=input_dim)
    epochs = args.i
    lamb = args.l
    lamb_N = lamb / N
    epsilon = args.epi

    for e in range(0, epochs):
        for i in range(0, N):
            ix = np.random.randint(0, N)
            peptide = peptides[ix]
            y_target = y[ix]
            y_pred = np.dot(peptide, weights)
            gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon)
        gerr, mse = cumulative_error(peptides, y, lamb, weights)
        train_pred = predict(peptides, weights)
        train_mse = cal_mse(y, train_pred)
        train_pcc = pearsonr(y, train_pred)
        eval_pred = predict(evaluation_peptides, weights)
        eval_mse = cal_mse(evaluation_targets, eval_pred)
        eval_pcc = pearsonr(evaluation_targets, eval_pred)
        print("Epoch: ", e, "Gerr:", gerr, train_pcc[0], train_mse, eval_pcc[0], eval_mse)

    matrix = vector_to_matrix(weights, alphabet)
    to_psi_blast(matrix)

if __name__ == "__main__":
    main()

# python smm_grad_descent.py -t /Users/anaselyoussef/Desktop/algo/data/SMM/A0201_training -e /Users/anaselyoussef/Desktop/algo/data/SMM/A0201_evaluation -l 0.01 -epi 0.05 -s 1 -i 100

# python smm_grad_descent.py -t /Users/anaselyoussef/Desktop/algo/SMM/A0201/f000 -e /Users/anaselyoussef/Desktop/algo/SMM/A0201/c000 | grep -v "#" > A0201.gd_mat.0