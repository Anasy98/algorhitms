import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Smith-Waterman Alignment')
parser.add_argument('-q', '--query', type=str, required=True, help='File with query sequence')
parser.add_argument('-db', '--database', type=str, required=True, help='File with database sequence')
parser.add_argument('-go', '--gap_open', type=float, default=-11.0, help='Value of gap open')
parser.add_argument('-ge', '--gap_extension', type=float, default=-1.0, help='Value of gap extension')
args = parser.parse_args()

data_dir = "/Users/anaselyoussef/Desktop/algo/data/"

gap_open = args.gap_open
gap_extension = args.gap_extension

#alphabet_file = "https://raw.githubusercontent.com/brunoalvarez89/data/master/algorithms_in_bioinformatics/part_3/alphabet"
alphabet_file = data_dir + "Matrices/alphabet"
alphabet = np.loadtxt(alphabet_file, dtype=str)

alphabet

#blosum_file = "https://raw.githubusercontent.com/brunoalvarez89/data/master/algorithms_in_bioinformatics/part_3/blosum50"
blosum_file = data_dir + "Matrices/BLOSUM50"

_blosum50 = np.loadtxt(blosum_file, dtype=float).reshape((24, -1)).T
blosum50 = {}

for i, letter_1 in enumerate(alphabet):
    
    blosum50[letter_1] = {}

    for j, letter_2 in enumerate(alphabet):
        
        blosum50[letter_1][letter_2] = _blosum50[i, j]

blosum50

def smith_waterman_alignment(query="VLLP", database="VLILP", scoring_scheme={}, gap_open=-5, gap_extension=-1):
    
    # Matrix dimensions
    M = len(query)
    N = len(database)
    
    # E matrix (for backtracking)
    E_matrix = np.zeros((M+1, N+1), dtype=object)
    
    # D matrix (alignment matrix)
    D_matrix = np.zeros((M+1, N+1), int)

    # Initialize matrices (Here you might add values to penalize end gaps)
    for i in range(M, 0, -1):
        D_matrix[i-1, N] = 0
        E_matrix[i-1, N] = 0

    for j in range(N, 0, -1):
        D_matrix[M, j-1] = 0
        E_matrix[M, j-1] = 0
    
    
    D_matrix_max_score, D_matrix_i_max, D_matrix_j_max = -9, -9, -9
    for i in range(M-1, -1, -1): 
        for j in range(N-1, -1, -1):
                
            # diagonal score
            diagonal_score = D_matrix[i+1, j+1] + scoring_scheme[query[i]][database[j]]
            # horizontal score
            # Gap opening
            max_horizontal_score = D_matrix[i, j+1] + gap_open
          
            # Gap extensions
            for k in range(j+2, N):
                score = D_matrix[i, k] + gap_open + (k - j) * gap_extension
                if score > max_horizontal_score: 
                    max_horizontal_score = score 
            
            
            # vertical score
            # Gap opening
            max_vertical_score = D_matrix[i+1, j] + gap_open
            
            # Gap extensions
            for k in range(i+2, M):
                score = D_matrix[k, j] + gap_open + (k - i) * gap_extension
                if score > max_vertical_score: 
                    max_vertical_score = score 
                  
            ####################
            # E_matrix entries #
            ####################
            # E[i,j] = 0, negative number
            # E[i,j] = 1, match
            # E[i,j] = 2, gap opening in database
            # E[i,j] = 3, gap extension in database
            # E[i,j] = 4, gap opening in query
            # E[i,j] = 5, gap extension in query
            
            if diagonal_score >= max_vertical_score and diagonal_score >= max_horizontal_score:
                max_score = diagonal_score
                direction = "diagonal"
            elif max_horizontal_score > max_vertical_score:
                max_score = max_horizontal_score
                direction = "horizontal"
            else:
                max_score = max_vertical_score
                direction = "vertical"
                
            if max_score <= 0:
                max_score = 0
                direction = "none"

            # diagonal direction case
            if direction == "diagonal":
                E_matrix[i,j] = 1
                
            # vertical direction case
            elif direction == "vertical":
                E_matrix[i,j] = 2 if max_vertical_score == D_matrix[i+1, j] + gap_open else 3
                        
            # horizontal direction case
            elif direction == "horizontal":
                E_matrix[i,j] = 4 if max_horizontal_score == D_matrix[i, j+1] + gap_open else 5

            else:
                # max_score is negative, put E to zero
                E_matrix[i,j] = 0
                 
            # store max score
            D_matrix[i, j] = max_score
            
            # fetch global max score
            if max_score > D_matrix_max_score:
                D_matrix_max_score = max_score
                D_matrix_i_max = i
                D_matrix_j_max = j
            
    return D_matrix, E_matrix, D_matrix_i_max, D_matrix_j_max, D_matrix_max_score


def smith_waterman_traceback(E_matrix, D_matrix, i_max, j_max, query="VLLP", database="VLILP", gap_open=-5, gap_extension=-1):
    
    M = len(query)
    N = len(database)
    
    aligned_query = []
    aligned_database = []
    positions = []
    matches = 0
    
    # start from max_i, max_j
    i, j = i_max, j_max
    while i < M and j < N :

        positions.append([i,j])
        
        # E[i,j] = 0, stop back tracking
        if E_matrix[i, j] == 0:
            break
        
        # E[i,j] = 1, match
        if E_matrix[i, j] == 1:
            aligned_query.append(query[i])
            aligned_database.append(database[j])
            if (query[i] == database[j]):
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

            # Find length of gap (check if score == D_matrix[i, j])
            while((score - D_matrix[i, j])*(score - D_matrix[i, j]) >= 0.00001): 
                count += 1
                score = D_matrix[count, j] + gap_open + (count-i-1) * gap_extension

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
            
            # Find length of gap (check if score == D_matrix[i, j])
            while((score - D_matrix[i, j])*(score - D_matrix[i, j]) >= 0.0001): 
                count += 1
                score = D_matrix[i, count] + gap_open + (count-j-1)*gap_extension

            for k in range(j, count):
                aligned_query.append("-")
                aligned_database.append(database[j])
                j += 1
                

    return aligned_query, aligned_database, matches


#Slides example
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

D_matrix, E_matrix, i_max, j_max, max_score = smith_waterman_alignment(query, database, scoring_scheme, gap_open, gap_extension)
aligned_query, aligned_database, matches = smith_waterman_traceback(E_matrix, D_matrix, i_max, j_max, query, database, gap_open, gap_extension)

print("ALN", query, len(query), database, len(database), len(aligned_query), matches, max_score)
print("QAL", ''.join(aligned_query))
print("DAL", ''.join(aligned_database))
print("")

print("---")

print("D Matrix")
print(D_matrix)
print("")
print("E Matrix")
print(E_matrix)

# python smith_waterman_o3.py -q /Users/anaselyoussef/Desktop/algo/data/Align/1PLB._.tab -db /Users/anaselyoussef/Desktop/algo/data/Align/database_list.tab -go -2 -ge -1

