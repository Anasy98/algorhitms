import numpy as np
from time import time

data_dir = "/Users/anaselyoussef/Desktop/algo/data/"

# File with formatted similarity scores
alignment_file_part = data_dir + "Hobohm/alignment_aln.fmt"

alignment_output = np.loadtxt(alignment_file_part, dtype=str)

def get_IDlist(alignment_output):
    IDlist = []
    first = 1

    for row in alignment_output:
        # The first entry is only included as query
        if first:
            id = row[0]
            first = 0
            if id not in IDlist:
                IDlist.append(id)

        id = row[1]

        if id not in IDlist:
            IDlist.append(id)

    return IDlist

t0 = time()

ID_list = get_IDlist(alignment_output)

nid_list = len(ID_list)

print("NID:", nid_list)
neighbor_matrix = np.zeros(shape=(nid_list, nid_list))

for i in range(nid_list):
    neighbor_matrix[i][i] = 1

for row in alignment_output:
    query_id = row[0]
    database_id = row[1]
    match = int(row[2])  # Match flag (0 or 1)
    
    ix = ID_list.index(query_id)
    iy = ID_list.index(database_id)
    
    if match == 1:
        neighbor_matrix[ix][iy] = 1
        neighbor_matrix[iy][ix] = 1

used = np.zeros(nid_list)

left = 1

while left > 0:
    max_nn = -99
    n_max = -9
    
    for i in range(nid_list):
        if used[i] == 0:
            nn = 0
            for j in range(nid_list):
                if used[j] == 0 and neighbor_matrix[i][j] == 1:
                    nn += 1
            
            if nn > max_nn:
                max_nn = nn
                n_max = i
    
    print("# Remove", max_nn, n_max, ID_list[n_max])
    if max_nn == 1:
        left = 0
    else:
        used[n_max] = 1
        
t1 = time()
print("Elapsed time (m):", (t1-t0)/60)
        
ncl = 0
for i in range(nid_list):
    if used[i] == 0:
        print("Unique", ID_list[i], ncl)
        ncl += 1
        
print("Number of unique sequences:", ncl)
