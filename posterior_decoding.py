import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_sequence', type=str, help='Input sequence or file path')
parser.add_argument('-s', '--state', type=int, default=2, help='State')
args = parser.parse_args()

def encode(sequence, symbols):
    enc = [0] * len(sequence)
    for i in range(len(sequence)):
        enc[i] = symbols.find(sequence[i])
    return enc

states = args.state
symbols = "123456"

# Check if the input_sequence is a file path by verifying if the path exists
if os.path.exists(args.input_sequence):
    with open(args.input_sequence, 'r') as file:
        input_sequence = file.read().strip()
else:
    input_sequence = args.input_sequence

input_encode = encode(input_sequence, symbols)

initial_prob = [1.0/states, 1.0/states]

transition_matrix = np.asarray([0.95, 0.05, 0.1, 0.9]).reshape(2, 2)

fair_prob = [1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6]
loaded_prob = [1.0/10, 1.0/10, 1.0/10, 1.0/10, 1.0/10, 5.0/10]
emission_probs = [fair_prob, loaded_prob]

def initialize_forward(input_encode, states, initial_prob, emission_probs):
    alpha = np.zeros(shape=(states, len(input_encode)))
    for i in range(0, states):
        alpha[i][0] = initial_prob[i] * emission_probs[i][input_encode[0]]
    return alpha

alpha = initialize_forward(input_encode, states, initial_prob, emission_probs)

# main loop
for i in range(1, len(input_encode)):
    for j in range(0, states):
        _sum = 0
        for k in range(0, states):
            _sum += alpha[k][i-1] * transition_matrix[k, j]
        # store prob
        alpha[j][i] = _sum * emission_probs[j][input_encode[i]]

print(alpha)

def initialize_backward(input_encode, states):
    beta = np.zeros(shape=(states, len(input_encode)))
    for i in range(0, states):
        beta[i][-1] = 1
    return beta

beta = initialize_backward(input_encode, states)

# main loop
for i in range(len(input_encode) - 2, -1, -1):
    for j in range(0, states):
        _sum = 0
        for k in range(0, states):
            _sum += emission_probs[k][input_encode[i+1]] * beta[k][i+1] * transition_matrix[j][k]
        # store prob
        beta[j][i] = _sum

print(beta)

# posterior = f * b / p_x
posterior = np.zeros(shape=(len(input_encode)), dtype=float)

p_state = 0

p_x = 0
for j in range(0, states):
    p_x += alpha[j][-1] * beta[j][-1]

print("Log(Px):", np.log(p_x))

for i in range(0, len(input_encode)):
    posterior[i] = (alpha[p_state][i] * beta[p_state][i]) / p_x
    print("Posterior", i, input_sequence[i], input_encode[i], np.log(alpha[p_state][i]), np.log(beta[p_state][i]), posterior[i])

# python posterior_decoding.py -i 34512331245366664666563266 -s 2

# python posterior_decoding.py -i /Users/anaselyoussef/Desktop/algo/data/HMM/casino.seqlong -s 2
