import numpy as np
import random
from scipy.stats import pearsonr
import argparse

data_dir = "/Users/anaselyoussef/Desktop/algo/data/"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lambd", type=float, default=1, help="Lambda (default: 1)")
    parser.add_argument("-t", "--training_file", type=str, required=True, help="File with training data")
    parser.add_argument("-e", "--evaluation_file", type=str, required=True, help="File with evaluation data")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Seed for random numbers (default: 1)")
    parser.add_argument("-i", "--iters", type=int, default=1000, help="Number of epochs to train (default: 1000)")
    parser.add_argument("-Ts", "--t_start", type=float, default=0.01, help="Start Temp (default: 0.01)")
    parser.add_argument("-Te", "--t_end", type=float, default=0.000001, help="End Temp (default: 0.000001)")
    parser.add_argument("-nT", "--t_steps", type=int, default=10, help="Number of T steps (default: 10)")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    T_i = args.t_start
    T_f = args.t_end
    T_steps = args.t_steps
    T_delta = (T_f - T_i) / T_steps
    T = np.linspace(T_i, T_f, T_steps)
    
    iters = args.iters
    lamb = args.lambd
    training_file = args.training_file
    evaluation_file = args.evaluation_file
    np.random.seed(args.seed)

    training = np.loadtxt(training_file, dtype=str)
    evaluation = np.loadtxt(evaluation_file, dtype=str)
    
    alphabet_file = data_dir + "Matrices/alphabet"
    alphabet = np.loadtxt(alphabet_file, dtype=str)

    sparse_file = data_dir + "Matrices/sparse"
    _sparse = np.loadtxt(sparse_file, dtype=float)
    sparse = {}

    for i, letter_1 in enumerate(alphabet):
        sparse[letter_1] = {}
        for j, letter_2 in enumerate(alphabet):
            sparse[letter_1][letter_2] = _sparse[i, j]

    def encode(peptides, encoding_scheme, alphabet):
        encoded_peptides = []
        for peptide in peptides:
            encoded_peptide = []
            for peptide_letter in peptide:
                for alphabet_letter in alphabet:
                    encoded_peptide.append(encoding_scheme[peptide_letter][alphabet_letter])
            encoded_peptides.append(encoded_peptide)
        return np.array(encoded_peptides)

    def cumulative_error(peptides, y, lamb, weights):
        error = 0
        for i in range(0, len(peptides)):
            peptide = peptides[i]
            y_target = y[i]
            y_pred = np.dot(peptide, weights)
            error += 1.0/2 * (y_pred - y_target)**2
        gerror = error + lamb * np.dot(weights, weights)
        error /= len(peptides)
        return gerror, error

    def predict(peptides, weights):
        pred = []
        for i in range(0, len(peptides)):
            peptide = peptides[i]
            y_pred = np.dot(peptide, weights)
            pred.append(y_pred)
        return pred

    def cal_mse(vec1, vec2):
        mse = 0
        for i in range(0, len(vec1)):
            mse += (vec1[i] - vec2[i])**2
        mse /= len(vec1)
        return mse

    # peptides
    peptides = training[:, 0]
    peptides = encode(peptides, sparse, alphabet)

    # target values
    y = np.array(training[:, 1], dtype=float)

    # evaluation peptides
    evaluation_peptides = evaluation[:, 0]
    evaluation_peptides = encode(evaluation_peptides, sparse, alphabet)

    # evaluation targets
    evaluation_targets = np.array(evaluation[:, 1], dtype=float)

    # weights
    input_dim = len(peptides[0])
    output_dim = 1
    w_bound = 0.1
    weights = np.random.uniform(-w_bound, w_bound, size=input_dim)

    perturbation_value = 0.1
    scale = 1.0/100

    number_of_tries = 0
    number_of_accepted = 0

    # calculate initial error
    gerror_initial, mse = cumulative_error(peptides, y, lamb, weights)

    # for each temperature
    for t in T:
        # for each iteration
        for i in range(0, iters):
            # get 2 random weight indexes
            weight_index_1 = np.random.randint(len(weights))
            weight_index_2 = np.random.randint(len(weights))

            # ensure they are different
            while weight_index_1 == weight_index_2:
                weight_index_2 = np.random.randint(len(weights))

            # store original weight values
            original_weight_1 = weights[weight_index_1]
            original_weight_2 = weights[weight_index_2]

            # apply random perturbation to both weights
            perturbation = np.random.uniform(0, perturbation_value)
            weights[weight_index_1] += perturbation
            weights[weight_index_2] -= perturbation

            # calculate new error
            gerror_new, mse = cumulative_error(peptides, y, lamb, weights)

            # compute error difference
            de = (gerror_new - gerror_initial) * scale

            # compute acceptance probability
            if de < 0:
                p = 1
            else:
                p = np.exp(-de/t)

            # throw coin
            coin = np.random.uniform(0.0, 1.0, 1)[0]

            # weight change is accepted
            if coin < p:
                gerror_initial = gerror_new
                number_of_accepted += 1
            # weight change is declined, restore previous weights
            else:
                weights[weight_index_1] = original_weight_1
                weights[weight_index_2] = original_weight_2

            number_of_tries += 1

            # define size of move so that on avarage 50% are accepted
            if number_of_tries == 100:
                if float(number_of_accepted)/number_of_tries > 0.5:
                    perturbation_value *= 1.1
                else:
                    perturbation_value *= 0.9
                number_of_tries = 0
                number_of_accepted = 0

            # predict on training data
            train_pred = predict(peptides, weights)
            train_mse = cal_mse(y, train_pred)
            train_pcc = pearsonr(y, train_pred)[0]

            # predict on evaluation data
            eval_pred = predict(evaluation_peptides, weights)
            eval_mse = cal_mse(evaluation_targets, eval_pred)
            eval_pcc = pearsonr(evaluation_targets, eval_pred)[0]

        print("t:", t, gerror_new, perturbation_value, train_mse, train_pcc, eval_mse, eval_pcc)

    def vector_to_matrix(vector, alphabet):
        rows = int(len(vector)/len(alphabet))
        matrix = [0] * rows
        offset = 0
        for i in range(0, rows):
            matrix[i] = {}
            for j in range(0, 20):
                matrix[i][alphabet[j]] = vector[j+offset]
            offset += len(alphabet)
        return matrix

    def to_psi_blast(matrix):
        header = ["", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*header))
        letter_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        for i, row in enumerate(matrix):
            scores = []
            scores.append(str(i+1) + " A")
            for letter in letter_order:
                score = row[letter]
                scores.append(round(score, 4))
            print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*scores))

    matrix = vector_to_matrix(weights, alphabet)
    to_psi_blast(matrix)

if __name__ == "__main__":
    main()


# python smm_m_carlo.py -l 1.5 -t /Users/anaselyoussef/Desktop/algo/data/SMM/A2403_training -e /Users/anaselyoussef/Desktop/algo/data/SMM/A2403_evaluation -s 42 -i 1500 -Ts 0.02 -Te 0.00001 -nT 15

# python smm_m_carlo.py -t /Users/anaselyoussef/Desktop/algo/SMM/A0201/f000 -e /Users/anaselyoussef/Desktop/algo/SMM/A0201/c000 | grep -v "#" > A0201.mc_mat.0