import numpy as np
import argparse

# Command line argument parsing
parser = argparse.ArgumentParser(description='Smith-Waterman sequence alignment')
parser.add_argument('-q', '--query_file', required=True, help='File with query sequence')
parser.add_argument('-db', '--db_file', required=True, help='File with database sequence')
parser.add_argument('-go', '--gap_open', type=float, default=-11.0, help='Value of gap open (-11.0)')
parser.add_argument('-ge', '--gap_extension', type=float, default=-1.0, help='Value of gap extension (-1.0)')
args = parser.parse_args()

data_dir = "/Users/anaselyoussef/Desktop/algo/data/"

gap_open = args.gap_open
gap_extension = args.gap_extension

alphabet_file = data_dir + "Matrices/alphabet"
alphabet = np.loadtxt(alphabet_file, dtype=str)

alphabet

blosum_file = data_dir + "Matrices/BLOSUM50"
_blosum50 = np.loadtxt(blosum_file, dtype=float).reshape((24, -1)).T

blosum50 = {}

for i, letter_1 in enumerate(alphabet):
    
    blosum50[letter_1] = {}

    for j, letter_2 in enumerate(alphabet):
        
        blosum50[letter_1][letter_2] = _blosum50[i, j]

blosum50

def smith_waterman_alignment(query="VLLP", database="VLILP", scoring_scheme={}, gap_open=-5, gap_extension=-1):

    # Matrix imensions
    M = len(query)
    N = len(database)
    
    # D matrix change to float
    D_matrix = np.zeros((M+1, N+1), int)

    # P matrix
    P_matrix = np.zeros((M+1, N+1), int)
    
    # Q matrix
    Q_matrix = np.zeros((M+1, N+1), int)

    # E matrix
    E_matrix = np.zeros((M+1, N+1), dtype=object)

    # Initialize matrices
    for i in range(M, 0, -1):
        # Here you might include  penalties for end gaps, i.e
        # alignment_matrix[i-1, N] = alignment_matrix[i, N] + gap_open
        D_matrix[i-1, N] = 0
        P_matrix[i-1, N] = 0
        Q_matrix[i-1, N] = 0
        E_matrix[i-1, N] = 0

    for j in range(N, 0, -1):
        # Here you might include  penalties for end gaps, i.e
        #alignment_matrix[M, j-1] = alignment_matrix[M, j] + gap_open
        D_matrix[M, j-1] = 0
        P_matrix[M, j-1] = 0
        Q_matrix[M, j-1] = 0
        E_matrix[M, j-1] = 0

        
    # Main loop
    D_matrix_max_score, D_matrix_i_max, D_matrix_j_max = -9, -9, -9
    for i in range(M-1, -1, -1): 
        for j in range(N-1, -1, -1):
            
            # Q_matrix[i,j] entry
            gap_open_database = D_matrix[i+1, j] + gap_open
            gap_extension_database = Q_matrix[i+1, j] + gap_extension
            max_gap_database = max(gap_open_database, gap_extension_database)
            
            Q_matrix[i,j] = max_gap_database
                
            # P_matrix[i,j] entry
            gap_open_query = D_matrix[i,j+1] + gap_open
            gap_extension_query = P_matrix[i,j+1] + gap_extension
            max_gap_query = max(gap_open_query, gap_extension_query)
            
            P_matrix[i,j] = max_gap_query
            
            # D_matrix[i,j] entry
            diagonal_score = D_matrix[i+1, j+1] + scoring_scheme[query[i]][database[j]]           
            
            # E_matrix[i,j] entry
            candidates = [(1, diagonal_score),
                          (2, gap_open_database),
                          (4, gap_open_query),
                          (3, gap_extension_database),
                          (5, gap_extension_query)]
            
            direction, max_score = max(candidates, key=lambda x: x[1])
            
            # check entry sign
            if max_score > 0:
                E_matrix[i,j] = direction
                D_matrix[i, j] = max_score
            else:
                E_matrix[i,j] = 0
                D_matrix[i, j] = 0

            # fetch global max score
            if max_score > D_matrix_max_score:
                D_matrix_max_score = max_score
                D_matrix_i_max = i
                D_matrix_j_max = j
            
    return P_matrix, Q_matrix, D_matrix, E_matrix, D_matrix_i_max, D_matrix_j_max, D_matrix_max_score

def smith_waterman_traceback(E_matrix, D_matrix, i_max, j_max, query="VLLP", database="VLILP", gap_open=-5, gap_extension=-1):
    
    M = len(query)
    N = len(database)
    
    aligned_query = []
    aligned_database = []
    
    # start from max_i, max_j
    i, j = i_max, j_max
    matches = 0
    while i < M and j < N:

        # E[i,j] = 0, stop back tracking
        if E_matrix[i, j] == 0:
            break
        
        
        # E[i,j] = 1, match
        if E_matrix[i, j] == 1:
            aligned_query.append(query[i])
            aligned_database.append(database[j])
            if ( query[i] == database[j]):
                matches += 1
            i += 1
            j += 1
        
        
        # E[i,j] = 2, gap opening in database
        if E_matrix[i, j] == 2:
            aligned_database.append("-")
            aligned_query.append(query[i])
            i += 1

            
        # E[i,j] = 3, gap extension in database
        if E_matrix[i, j] == 3:
                   
            count = i + 2
            score = D_matrix[count, j] + gap_open + gap_extension
            
            # Find length of gap
            while((score - D_matrix[i, j])*(score - D_matrix[i, j]) >= 0.00001):   
                count += 1
                score = D_matrix[count, j] + gap_open + (count-i-1)*gap_extension
                
            for k in range(i, count):
                aligned_database.append("-")
                aligned_query.append(query[i])
                i += 1
            
            
        # E[i,j] = 4, gap opening in query
        if E_matrix[i, j] == 4:
            aligned_query.append("-")
            aligned_database.append(database[j])
            j += 1
        
        
        # E[i,j] = 5, gap extension in query
        if E_matrix[i, j] == 5:
             
            count = j + 2
            score = D_matrix[i, count] + gap_open + gap_extension
            
            # Find length of gap
            while((score - D_matrix[i, j])*(score - D_matrix[i, j]) >= 0.0001): 
                count += 1
                score = D_matrix[i, count] + gap_open + (count-j-1)*gap_extension

            for k in range(j, count):
                aligned_query.append("-")
                aligned_database.append(database[j])
                j += 1

                
    return aligned_query, aligned_database, matches

# Morten's slides
#query = "VLLP"
#database = "VLILP"
#scoring_scheme = blosum50
#gap_open = -5
#gap_extension = -1

#Matrix dump exercise 2
query = "VLPVLILP"
database = "VLLPVLLP"
scoring_scheme = blosum50
gap_open = -2
gap_extension = -1

#Matrix dump exercise 1
#query = "IDVLLGADDGSLAFVPSEFSISPGEKIVFKNNAGFPHNIVFDEDSIPSGVDASKISMSEEDLLNAKGETFEVALSNKGEYSFYCSPHQGAGMVGKVTVN"
#database = "AEVKLGSDDGGLVFSPSSFTVAAGEKITFKNNAGFPHNIVFDEDEVPAGVNAEKISQPEYLNGAGETYEVTLTEKGTYKFYCEPHAGAGMKGEVTVN"
#scoring_scheme = blosum50
#gap_open = -11
#gap_extension = -1

P_matrix, Q_matrix, D_matrix, E_matrix, i_max, j_max, max_score = smith_waterman_alignment(query, database, scoring_scheme, gap_open, gap_extension)
aligned_query, aligned_database, matches = smith_waterman_traceback(E_matrix, D_matrix, i_max, j_max, query, database, gap_open, gap_extension)
    
print("ALN", "Query", len(query), "Database", len(database), len(aligned_query), max_score, matches)
print("QAL", ''.join(aligned_query))
print("DAL", ''.join(aligned_database))

print("---")

print("D Matrix")
print(D_matrix)
print("")
print("E Matrix")
print(E_matrix)
print("")
print("P Matrix")
print(P_matrix)
print("")
print("E Matrix")
print(E_matrix)


# python smith_waterman_o2.py -q /Users/anaselyoussef/Desktop/algo/data/Align/1PLB._.tab -db /Users/anaselyoussef/Desktop/algo/data/Align/database_list.tab -go -2 -ge -1
