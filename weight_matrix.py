import numpy as np
import math
from pprint import pprint
import argparse
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate a PSSM from peptide data.')
    parser.add_argument('-b', type=float, default=50.0, help='Weight on pseudo count (default: 50.0)')
    parser.add_argument('-w', action='store_true', help='Use sequence weighting')
    parser.add_argument('-f', type=str, required=True, help='File with peptides')
    return parser.parse_args()

def initialize_matrix(peptide_length, alphabet):
    init_matrix = [0] * peptide_length
    for i in range(0, peptide_length):
        row = {}
        for letter in alphabet:
            row[letter] = 0.0
        init_matrix[i] = row
    return init_matrix

def to_psi_blast(matrix):
    header = ["", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*header))
    letter_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    for i, row in enumerate(matrix):
        scores = []
        scores.append(str(i + 1) + " A")
        for letter in letter_order:
            score = row[letter]
            scores.append(f"{score:.3f}")
        print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*scores))

def main():
    args = parse_arguments()
    beta = args.b
    sequence_weighting = args.w
    peptides_file = args.f

    data_dir = "/Users/anaselyoussef/Desktop/algo/data/"

    alphabet_file = data_dir + "Matrices/alphabet"
    alphabet = np.loadtxt(alphabet_file, dtype=str)

    bg_file = data_dir + "Matrices/bg.freq.fmt"
    _bg = np.loadtxt(bg_file, dtype=float)

    bg = {}
    for i in range(0, len(alphabet)):
        bg[alphabet[i]] = _bg[i]

    blosum62_file = data_dir + "Matrices/blosum62.freq_rownorm"
    _blosum62 = np.loadtxt(blosum62_file, dtype=float).T

    blosum62 = {}
    for i, letter_1 in enumerate(alphabet):
        blosum62[letter_1] = {}
        for j, letter_2 in enumerate(alphabet):
            blosum62[letter_1][letter_2] = _blosum62[i, j]

    peptides_data = np.loadtxt(peptides_file, dtype=str)
    peptides = [row[0] for row in peptides_data]

    peptide_length = len(peptides[0])

    for i in range(0, len(peptides)):
        if len(peptides[i]) != peptide_length:
            print("Error, peptides differ in length!")
            return

    c_matrix = initialize_matrix(peptide_length, alphabet)

    for position in range(0, peptide_length):
        for peptide in peptides:
            c_matrix[position][peptide[position]] += 1

    pprint({k: f"{v:.3f}" for k, v in c_matrix[0].items()})

    weights = {}

    for peptide in peptides:
        if sequence_weighting:
            w = 0.0
            neff = 0.0
            for position in range(0, peptide_length):
                r = 0
                for letter in alphabet:
                    if c_matrix[position][letter] != 0:
                        r += 1
                s = c_matrix[position][peptide[position]]
                w += 1.0 / (r * s)
                neff += r
            neff = neff / peptide_length
        else:
            w = 1
            neff = len(peptides)
        weights[peptide] = w

    pprint("W:")
    pprint({k: f"{v:.3f}" for k, v in weights.items()})
    pprint("Nseq:")
    pprint(f"{neff:.3f}")

    f_matrix = initialize_matrix(peptide_length, alphabet)

    for position in range(0, peptide_length):
        n = 0
        for peptide in peptides:
            f_matrix[position][peptide[position]] += weights[peptide]
            n += weights[peptide]
        for letter in alphabet:
            f_matrix[position][letter] /= n

    pprint({k: f"{v:.3f}" for k, v in f_matrix[0].items()})

    g_matrix = initialize_matrix(peptide_length, alphabet)

    for position in range(0, peptide_length):
        for letter_1 in alphabet:
            for letter_2 in alphabet:
                g_matrix[position][letter_1] += f_matrix[position][letter_2] * blosum62[letter_1][letter_2]

    pprint({k: f"{v:.3f}" for k, v in g_matrix[0].items()})

    p_matrix = initialize_matrix(peptide_length, alphabet)

    alpha = neff - 1

    for position in range(0, peptide_length):
        for a in alphabet:
            p_matrix[position][a] = (alpha * f_matrix[position][a] + beta * g_matrix[position][a]) / (alpha + beta)

    pprint({k: f"{v:.3f}" for k, v in p_matrix[0].items()})

    w_matrix = initialize_matrix(peptide_length, alphabet)

    for position in range(0, peptide_length):
        for letter in alphabet:
            if p_matrix[position][letter] > 0:
                w_matrix[position][letter] = 2 * math.log(p_matrix[position][letter] / bg[letter]) / math.log(2)
            else:
                w_matrix[position][letter] = -999.900

    pprint({k: f"{v:.3f}" for k, v in w_matrix[0].items()})

    to_psi_blast(w_matrix)

    # Scoring peptides against the generated PSSM and calculating PCC
    evaluation_file = peptides_file  # Using the same file for evaluation
    evaluation_data = np.loadtxt(evaluation_file, dtype=str)
    evaluation_peptides = evaluation_data[:, 0]
    evaluation_targets = evaluation_data[:, 1].astype(float)

    def score_peptide(peptide, matrix):
        acum = 0
        for i in range(0, len(peptide)):
            acum += matrix[i][peptide[i]]
        return acum

    evaluation_predictions = []
    for evaluation_peptide in evaluation_peptides:
        evaluation_predictions.append(score_peptide(evaluation_peptide, w_matrix))

    pcc = pearsonr(evaluation_targets, evaluation_predictions)
    print(f"PCC: {pcc[0]:.3f}")

    plt.scatter(evaluation_targets, evaluation_predictions)
    plt.xlabel("Actual Targets")
    plt.ylabel("Predicted Scores")
    plt.title("Scatter plot of Actual vs Predicted Scores")
    plt.show()

if __name__ == "__main__":
    main()
