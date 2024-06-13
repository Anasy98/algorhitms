import numpy as np
import math
from Bio import motifs
from Bio.Seq import Seq
from argparse import ArgumentParser
from collections import Counter

def parse_arguments():
    parser = ArgumentParser(description="Pep2mat")
    parser.add_argument("-b", action="store", dest="beta", type=float, default=50.0, help="Weight on pseudo count (default: 50.0)")
    parser.add_argument("-w", action="store_true", dest="sequence_weighting", help="Use Sequence weighting")
    parser.add_argument("-f", action="store", dest="peptides_file", type=str, help="File with peptides", required=True)
    return parser.parse_args()

def read_peptides(file_path):
    with open(file_path) as file:
        peptides = [line.strip() for line in file]
    return peptides

def to_psi_blast(pssm):
    header = ["", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*header))
    letter_order = header[1:]
    for i, row in enumerate(pssm):
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

    peptides = read_peptides(peptides_file)
    
    # Check lengths of peptides
    peptide_lengths = [len(peptide) for peptide in peptides]
    print(f"Peptide lengths: {Counter(peptide_lengths)}")
    
    # Find the most common peptide length
    most_common_length = Counter(peptide_lengths).most_common(1)[0][0]
    filtered_peptides = [peptide for peptide in peptides if len(peptide) == most_common_length]
    
    if len(filtered_peptides) != len(peptides):
        print(f"Warning: {len(peptides) - len(filtered_peptides)} peptides were excluded due to inconsistent lengths.")

    instances = [Seq(peptide) for peptide in filtered_peptides]
    motif = motifs.create(instances)
    
    pwm = motif.counts.normalize(pseudocounts=beta)
    
    background = {letter: 1.0 / 20 for letter in "ARNDCQEGHILKMFPSTWYV"}  # Uniform background distribution
    pssm = pwm.log_odds(background=background)
    
    to_psi_blast(pssm)
    
    # Compare PSSMs using Pearson correlation coefficient (against itself for demonstration)
    distance, offset = pssm.dist_pearson(pssm)
    print(f"Pearson correlation distance = {distance:.3g}")
    print(f"Offset = {offset}")

if __name__ == "__main__":
    main()
