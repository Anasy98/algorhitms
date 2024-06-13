import numpy as np
import math
from pprint import pprint
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser(description="Pep2mat")
    parser.add_argument("-b", action="store", dest="beta", type=float, default=50.0, help="Weight on pseudo count (default: 50.0)")
    parser.add_argument("-w", action="store_true", dest="sequence_weighting", help="Use Sequence weighting")
    parser.add_argument("-f", action="store", dest="peptides_file", type=str, help="File with peptides", required=True)
    return parser.parse_args()

def initialize_matrix(peptide_length, alphabet):
    init_matrix = [0] * peptide_length
    for i in range(peptide_length):
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
        scores = [str(i + 1) + " A"]
        for letter in letter_order:
            score = row[letter]
            scores.append(f"{score:.3f}")
        print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*scores))

def main():
    args = parse_arguments()
    beta = args.beta
    sequence_weighting = args.sequence_weighting
    peptides_file = args.peptides_file

    data_dir = "/Users/anaselyoussef/Desktop/algo/data/"

    alphabet_file = data_dir + "Matrices/alphabet"
    alphabet = np.loadtxt(alphabet_file, dtype=str)

    bg_file = data_dir + "Matrices/bg.freq.fmt"
    _bg = np.loadtxt(bg_file, dtype=float)

    bg = {alphabet[i]: _bg[i] for i in range(len(alphabet))}

    blosum62_file = data_dir + "Matrices/blosum62.freq_rownorm"
    _blosum62 = np.loadtxt(blosum62_file, dtype=float).T

    blosum62 = {letter_1: {letter_2: _blosum62[i, j] for j, letter_2 in enumerate(alphabet)} for i, letter_1 in enumerate(alphabet)}

    peptides_data = np.loadtxt(peptides_file, dtype=str)
    peptides = [row[0] for row in peptides_data]

    peptide_length = len(peptides[0])

    for peptide in peptides:
        if len(peptide) != peptide_length:
            print("Error, peptides differ in length!")
            return

    c_matrix = initialize_matrix(peptide_length, alphabet)

    for position in range(peptide_length):
        for peptide in peptides:
            c_matrix[position][peptide[position]] += 1

    weights = {}
    neff = 0.0

    for peptide in peptides:
        if sequence_weighting:
            w = 0.0
            for position in range(peptide_length):
                r = sum(1 for letter in alphabet if c_matrix[position][letter] != 0)
                s = c_matrix[position][peptide[position]]
                w += 1.0 / (r * s)
                neff += r
            neff /= peptide_length
        else:
            w = 1
            neff = len(peptides)
        weights[peptide] = w

    f_matrix = initialize_matrix(peptide_length, alphabet)

    for position in range(peptide_length):
        n = sum(weights[peptide] for peptide in peptides)
        for peptide in peptides:
            f_matrix[position][peptide[position]] += weights[peptide]
        for letter in alphabet:
            f_matrix[position][letter] /= n

    g_matrix = initialize_matrix(peptide_length, alphabet)

    for position in range(peptide_length):
        for letter_1 in alphabet:
            for letter_2 in alphabet:
                g_matrix[position][letter_1] += f_matrix[position][letter_2] * blosum62[letter_1][letter_2]

    p_matrix = initialize_matrix(peptide_length, alphabet)
    alpha = neff - 1

    for position in range(peptide_length):
        for a in alphabet:
            p_matrix[position][a] = (alpha * f_matrix[position][a] + beta * g_matrix[position][a]) / (alpha + beta)

    w_matrix = initialize_matrix(peptide_length, alphabet)

    for position in range(peptide_length):
        for letter in alphabet:
            if p_matrix[position][letter] > 0:
                w_matrix[position][letter] = 2 * math.log(p_matrix[position][letter] / bg[letter]) / math.log(2)
            else:
                w_matrix[position][letter] = -999.900

    to_psi_blast(w_matrix)

if __name__ == "__main__":
    main()


# python pep2mat.py -b 50.0 -w -f /Users/anaselyoussef/Desktop/algo/data/PSSM/A0201.evalpep2mat.py -b 50.0 -w -f /Users/anaselyoussef/Desktop/algo/data/PSSM/A0201.eval