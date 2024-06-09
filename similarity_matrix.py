import numpy as np
import matplotlib.pyplot as plt

data_dir = "/Users/anaselyoussef/Desktop/algo/data/"

#alphabet_file = "https://raw.githubusercontent.com/brunoalvarez89/data/master/algorithms_in_bioinformatics/part_1/alphabet"
alphabet_file = data_dir + "Matrices/alphabet"
alphabet = np.loadtxt(alphabet_file, dtype=str)

#alphabet

#blosum50_file = "https://raw.githubusercontent.com/brunoalvarez89/data/master/algorithms_in_bioinformatics/part_1/blosum50"
blosum50_file = data_dir + "Matrices/blosum50"
_blosum50 = np.loadtxt(blosum50_file, dtype=float)

blosum50 = {}

for i, letter_1 in enumerate(alphabet):
    
    blosum50[letter_1] = {}

    for j, letter_2 in enumerate(alphabet):
        
        blosum50[letter_1][letter_2] = _blosum50[i, j]

#blosum50

#sequence_1_file = "https://raw.githubusercontent.com/brunoalvarez89/data/master/algorithms_in_bioinformatics/part_1/sequence_1"
sequence_1_file = data_dir + "Intro/1PLC._.tab"
sequence_1_id = np.loadtxt(sequence_1_file, dtype=str)[0]
sequence_1 = np.loadtxt(sequence_1_file, dtype=str)[1]

#sequence_1_id, sequence_1

#sequence_2_file = "https://raw.githubusercontent.com/brunoalvarez89/data/master/algorithms_in_bioinformatics/part_1/sequence_2"
sequence_2_file = data_dir + "Intro/1PLB._.tab"

sequence_2_id = np.loadtxt(sequence_2_file, dtype=str)[0]
sequence_2 = np.loadtxt(sequence_2_file, dtype=str)[1]

#sequence_2_id, sequence_2

score_matrix = np.zeros(shape=(len(sequence_1), len(sequence_2)))

M = score_matrix.shape[0]
N = score_matrix.shape[1]

for i in range(0, M):
    
    for j in range(0, N):
        
        score_matrix[i, j] = blosum50[sequence_1[i]][sequence_2[j]]

#score_matrix[0]

fig = plt.figure(figsize=(10, 10), dpi= 80)

plt.imshow(score_matrix, cmap="coolwarm")
plt.ylabel(sequence_1_id, fontsize=18);
plt.xlabel(sequence_2_id, fontsize=18);
plt.colorbar();
plt.show()