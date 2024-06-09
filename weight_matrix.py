import numpy as np
import math
from pprint import pprint
from argparse import ArgumentParser

parser = ArgumentParser(description="Pep2mat")

parser.add_argument("-b", action="store", dest="beta", type=float, default=50.0, help="Weight on pseudo count (default: 50.0)")
parser.add_argument("-w", action="store_true", dest="sequence_weighting", help="Use Sequence weighting")
parser.add_argument("-f", action="store", dest="peptides_file", type=str, help="File with peptides")
args = parser.parse_args()
beta = args.beta
sequence_weighting = args.sequence_weighting
peptides_file = args.peptides_file

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
        exit()

def initialize_matrix(peptide_length, alphabet):
    init_matrix = [0] * peptide_length
    for i in range(0, peptide_length):
        row = {}
        for letter in alphabet:
            row[letter] = 0.0
        init_matrix[i] = row
    return init_matrix

c_matrix = initialize_matrix(peptide_length, alphabet)

for position in range(0, peptide_length):
    for peptide in peptides:
        c_matrix[position][peptide[position]] += 1

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

f_matrix = initialize_matrix(peptide_length, alphabet)

for position in range(0, peptide_length):
    n = 0
    for peptide in peptides:
        f_matrix[position][peptide[position]] += weights[peptide]
        n += weights[peptide]
    for letter in alphabet:
        f_matrix[position][letter] /= n

g_matrix = initialize_matrix(peptide_length, alphabet)

for position in range(0, peptide_length):
    for letter_1 in alphabet:
        for letter_2 in alphabet:
            g_matrix[position][letter_1] += f_matrix[position][letter_2] * blosum62[letter_1][letter_2]

p_matrix = initialize_matrix(peptide_length, alphabet)

alpha = neff - 1

for position in range(0, peptide_length):
    for a in alphabet:
        p_matrix[position][a] = (alpha * f_matrix[position][a] + beta * g_matrix[position][a]) / (alpha + beta)

w_matrix = initialize_matrix(peptide_length, alphabet)

for position in range(0, peptide_length):
    for letter in alphabet:
        if p_matrix[position][letter] > 0:
            w_matrix[position][letter] = 2 * math.log(p_matrix[position][letter] / bg[letter]) / math.log(2)
        else:
            w_matrix[position][letter] = -999.900

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

to_psi_blast(w_matrix)

# python weight_matrix.py -b 50.0 -w -f /Users/anaselyoussef/Desktop/algo/data/PSSM/A0201.eval

# ### convert w_matrix to PSI-BLAST format and print to file

# Write out PSSM in Psi-Blast format to file
#file_name = "w_matrix_test"
#to_psi_blast_file(w_matrix, file_name)