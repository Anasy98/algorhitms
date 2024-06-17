import numpy as np
from time import time
import argparse
import os
import math

def parse_arguments():
    parser = argparse.ArgumentParser(description="Homology sequence alignment tool")
    parser.add_argument("-f", "--file", help="File input data", required=True)
    return parser.parse_args()

def load_alphabet_and_blosum(data_dir):
    alphabet_file = os.path.join(data_dir, "Matrices", "alphabet")
    alphabet = np.loadtxt(alphabet_file, dtype=str)

    blosum_file = os.path.join(data_dir, "Matrices", "BLOSUM50")
    _blosum50 = np.loadtxt(blosum_file, dtype=int).T

    blosum50 = {}
    for i, letter_1 in enumerate(alphabet):
        blosum50[letter_1] = {}
        for j, letter_2 in enumerate(alphabet):
            blosum50[letter_1][letter_2] = _blosum50[i, j]
    return alphabet, blosum50

def load_sequences(data_dir):
    database_file = os.path.join(data_dir, "Hobohm", "database_list.tab")
    database_list = np.loadtxt(database_file, dtype=str).reshape(-1, 2)

    ids = database_list[:, 0]
    sequences = database_list[:, 1]

    return sequences, ids

def smith_waterman(query, database, scoring_scheme, gap_open, gap_extension):
    P_matrix, Q_matrix, D_matrix, E_matrix, i_max, j_max, max_score = smith_waterman_alignment(query, database, scoring_scheme, gap_open, gap_extension)
    aligned_query, aligned_database, matches = smith_waterman_traceback(E_matrix, D_matrix, i_max, j_max, query, database, gap_open, gap_extension)
    return aligned_query, aligned_database, matches

def smith_waterman_alignment(query, database, scoring_scheme, gap_open, gap_extension):
    M = len(query)
    N = len(database)
    
    D_matrix = np.zeros((M+1, N+1), int)
    P_matrix = np.zeros((M+1, N+1), int)
    Q_matrix = np.zeros((M+1, N+1), int)
    E_matrix = np.zeros((M+1, N+1), dtype=object)

    D_matrix_max_score, D_matrix_i_max, D_matrix_j_max = -9, -9, -9
    for i in range(M-1, -1, -1):
        for j in range(N-1, -1, -1):
            gap_open_database = D_matrix[i+1, j] + gap_open
            gap_extension_database = Q_matrix[i+1, j] + gap_extension
            max_gap_database = max(gap_open_database, gap_extension_database)
            Q_matrix[i, j] = max_gap_database
                
            gap_open_query = D_matrix[i, j+1] + gap_open
            gap_extension_query = P_matrix[i, j+1] + gap_extension
            max_gap_query = max(gap_open_query, gap_extension_query)
            P_matrix[i, j] = max_gap_query
            
            diagonal_score = D_matrix[i+1, j+1] + scoring_scheme[query[i]][database[j]]
            candidates = [(1, diagonal_score),
                          (2, gap_open_database),
                          (4, gap_open_query),
                          (3, gap_extension_database),
                          (5, gap_extension_query)]
            
            direction, max_score = max(candidates, key=lambda x: x[1])
            if max_score > 0:
                E_matrix[i, j] = direction
            else:
                E_matrix[i, j] = 0
            if max_score > 0:
                D_matrix[i, j] = max_score
            else:
                D_matrix[i, j] = 0

            if max_score > D_matrix_max_score:
                D_matrix_max_score = max_score
                D_matrix_i_max = i
                D_matrix_j_max = j
            
    return P_matrix, Q_matrix, D_matrix, E_matrix, D_matrix_i_max, D_matrix_j_max, D_matrix_max_score

def smith_waterman_traceback(E_matrix, D_matrix, i_max, j_max, query, database, gap_open, gap_extension):
    M = len(query)
    N = len(database)
    aligned_query = []
    aligned_database = []
    matches = 0

    i, j = i_max, j_max
    while i < M and j < N:
        if E_matrix[i, j] == 0:
            break
        if E_matrix[i, j] == 1:
            aligned_query.append(query[i])
            aligned_database.append(database[j])
            if query[i] == database[j]:
                matches += 1
            i += 1
            j += 1
        elif E_matrix[i, j] == 2:
            aligned_database.append("-")
            aligned_query.append(query[i])
            i += 1
        elif E_matrix[i, j] == 3:
            count = i + 2
            score = D_matrix[count, j] + gap_open + gap_extension
            while (score - D_matrix[i, j]) * (score - D_matrix[i, j]) >= 0.00001:
                count += 1
                score = D_matrix[count, j] + gap_open + (count - i - 1) * gap_extension
            for k in range(i, count):
                aligned_database.append("-")
                aligned_query.append(query[i])
                i += 1
        elif E_matrix[i, j] == 4:
            aligned_query.append("-")
            aligned_database.append(database[j])
            j += 1
        elif E_matrix[i, j] == 5:
            count = j + 2
            score = D_matrix[i, count] + gap_open + gap_extension
            while (score - D_matrix[i, j]) * (score - D_matrix[i, j]) >= 0.0001:
                count += 1
                score = D_matrix[i, count] + gap_open + (count - j - 1) * gap_extension
            for k in range(j, count):
                aligned_query.append("-")
                aligned_database.append(database[j])
                j += 1

    return aligned_query, aligned_database, matches

def homology_function(alignment_length, matches):
    # Using the formula from the image: %Id > 290 / sqrt(alignment_length)
    threshold = 290 / math.sqrt(alignment_length)
    homology_score = (matches / alignment_length) * 100
    if homology_score > threshold:
        return "discard", homology_score
    else:
        return "keep", homology_score

def main():
    args = parse_arguments()
    data_dir = args.file

    alphabet, blosum50 = load_alphabet_and_blosum(data_dir)
    candidate_sequences, candidate_ids = load_sequences(data_dir)
    print("# Number of elements:", len(candidate_sequences))

    accepted_sequences, accepted_ids = [], []
    accepted_sequences.append(candidate_sequences[0])
    accepted_ids.append(candidate_ids[0])
    print("# Unique.", 0, len(accepted_sequences)-1, accepted_ids[0])

    scoring_scheme = blosum50
    gap_open = -11
    gap_extension = -1

    t0 = time()

    for i in range(1, len(candidate_sequences)):
        for j in range(0, len(accepted_sequences)):
            query = candidate_sequences[i]
            database = accepted_sequences[j]
            aligned_query, aligned_database, matches = smith_waterman(query, database, scoring_scheme, gap_open, gap_extension)
            alignment_length = len(aligned_query)
            homology_outcome, homology_score = homology_function(alignment_length, matches)
            if homology_outcome == "discard":
                print("# Not unique.", i, candidate_ids[i], "is homolog to", accepted_ids[j], homology_score)
                break
        if homology_outcome == "keep":
            accepted_sequences.append(candidate_sequences[i])
            accepted_ids.append(candidate_ids[i])
            print("# Unique.", i, len(accepted_sequences)-1, candidate_ids[i], homology_score)

    t1 = time()
    print("Elapsed time (m):", (t1 - t0) / 60)
    print("Accepted sequences:", len(accepted_ids))
    for i in range(len(accepted_ids)):
        print(accepted_ids[i])

if __name__ == "__main__":
    main()

#python Hobohm1.py -f /Users/anaselyoussef/Desktop/algo/data
