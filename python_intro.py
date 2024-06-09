import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser(description="A very first Python program")
parser.add_argument("-t", action="store", dest="threshold", type=float, default=0.5, help="Target value filtering threshold (default: 0.5)")
parser.add_argument("-f", action="store", dest="peptides_targets_file", type=str, help="Peptides-Targets file")
parser.add_argument("-l", action="store", dest="peplen_threshold", type=int, default=9,help="Peptide length (default: 9)")

args = parser.parse_args()
threshold = args.threshold
peptides_targets_file = args.peptides_targets_file
peplen_threshold = args.peplen_threshold


data_dir = "/Users/anaselyoussef/Desktop/algo/data/"

#peptides_targets_file = data_dir + "Intro/test.dat"

peptides_targets = np.loadtxt(peptides_targets_file, dtype=str).reshape(-1,2)

print(peptides_targets.shape)

peptides = peptides_targets[:, 0]

print(type(peptides), type(peptides_targets))

targets = peptides_targets[:, 1].astype(float)

peptides_9mer = []
targets_9mer = []

for i in range(0, len(peptides)):
    
    if len(peptides[i]) == peplen_threshold:
        
        peptides_9mer.append(peptides[i])
        
        targets_9mer.append(targets[i])

#threshold = 0.5

to_remove = []

for i in range(0, len(peptides_9mer)):
        
        if targets_9mer[i] < threshold:

            to_remove.append(i)


peptides_9mer_t = np.delete(peptides_9mer, to_remove)
targets_9mer_t = np.delete(targets_9mer, to_remove)

error = False

for i in range(0, len(peptides_9mer_t)):
        
        if targets_9mer_t[i] < threshold:

            error = True
            
            break

if error:

    print("Something went wrong")
    
else:
    
    print("Success")


for i in range(0, len(peptides_9mer_t)):
    
    print(peptides_9mer_t[i], targets_9mer_t[i])

# python python_intro.py -t 0.5 -f /Users/anaselyoussef/Desktop/algo/data/Intro/test.dat -l 9