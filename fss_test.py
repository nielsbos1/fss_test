from fill_sketch import FillSketch
import numpy as np
import math


def get_seeds():
    with open('seeds.txt', 'r') as file_:
        lines = file_.readlines()
    output_list = []
    for line in lines:
        output_list.append(int(line))
    return output_list

seeds = get_seeds()

# Main experiment
# Use the FillSketch class to estimate the jaccard similarity between two sets
sketch_length = 16
no_estimations = 10000
array_estimations = np.empty(shape=no_estimations)
for index, seed in enumerate(seeds):
    if index >= no_estimations:
        break
    input_1 = ['1', '2']
    input_2 = ['1', '3']

    fill_sketch_1 = FillSketch(input=input_1, sketch_length=16, seed=seed)
    fill_sketch_2 = FillSketch(input=input_2, sketch_length=16, seed=seed)

    jaccard = fill_sketch_1.get_estimated_jaccard_similarity(fill_sketch_2)
    array_estimations[index] = jaccard

print(f"Average Jaccard similarity: {array_estimations.mean()}")

def generate_fill_sketch(input_set, sketch_length, hash_outputs):
    sketch = np.repeat(math.inf, repeats=sketch_length)
    c = 0
    for i in range(2 * sketch_length):
        for input in input_set:
            if i < sketch_length:
                bin_value = int(hash_outputs[input][i])
                v_value = i
            else:
                bin_value = i - sketch_length
                v_value = i
            if math.isinf(sketch[bin_value]):
                c += 1
            sketch[bin_value] = min(sketch[bin_value], v_value)
        if c == sketch_length:
            return sketch
    return sketch

# Experiment 1
# Random hash function is implemented by taking the value associated to the first index of a random permuation of 16
array_estimations_experiment_1 = np.empty(shape=no_estimations)
for i in range(no_estimations):
    np.random.RandomState(seeds[i])
    permutation_dictionary = {0: [], 1: [], 2: []}
    # assign the 
    for j in range(sketch_length):
        permutation_dictionary[0].append(np.random.permutation(16))
    for j in range(sketch_length):
        permutation_dictionary[1].append(np.random.permutation(16))
    for j in range(sketch_length):
        permutation_dictionary[2].append(np.random.permutation(16))
    input_1 = [0, 1]
    input_2 = [0, 2]
    hash_outputs_1 = {0: [], 1: []}
    hash_outputs_2 = {0: [], 2: []}
    for j in range(sketch_length):
        hash_outputs_1[0].append(permutation_dictionary[0][j][0])
        hash_outputs_1[1].append(permutation_dictionary[1][j][0])
        hash_outputs_2[0].append(permutation_dictionary[0][j][0])
        hash_outputs_2[2].append(permutation_dictionary[2][j][0])

    sketch_1 = generate_fill_sketch(input_1, 16, hash_outputs_1)
    sketch_2 = generate_fill_sketch(input_2, 16, hash_outputs_2)

    # compute jaccard
    counter = 0
    for k in range(sketch_length):
        if sketch_1[k] == sketch_2[k]:
            counter += 1
    jaccard = counter / sketch_length
    array_estimations_experiment_1[i] = jaccard

print(f"Average Jaccard similarity: {array_estimations_experiment_1.mean()}")

# EXPERIMENT 2
# Random hash function is implemented by taking the assigning the numbers on the first three indices of a random permuation of 16 as the hash outputs
# Set the hash output
array_estimations_experiment_2 = np.empty(shape=no_estimations)
for i in range(no_estimations):
    np.random.RandomState(seed=seeds[i])
    permutation_list = []
    for j in range(sketch_length):
        permutation_list.append(np.random.permutation(16))
    input_1 = [0, 1]
    input_2 = [0, 2]
    hash_outputs_1 = {0: [], 1: []}
    hash_outputs_2 = {0: [], 2: []}
    for j in range(sketch_length):
        hash_outputs_1[0].append(permutation_list[j][0])
        hash_outputs_1[1].append(permutation_list[j][1])
        hash_outputs_2[0].append(permutation_list[j][0])
        hash_outputs_2[2].append(permutation_list[j][2])

    sketch_1 = generate_fill_sketch(input_1, 16, hash_outputs_1)
    sketch_2 = generate_fill_sketch(input_2, 16, hash_outputs_2)

    # compute jaccard
    counter = 0
    for k in range(sketch_length):
        if sketch_1[k] == sketch_2[k]:
            counter += 1
    jaccard = counter / sketch_length
    array_estimations_experiment_2[i] = jaccard

print(f"Average Jaccard similarity: {array_estimations_experiment_2.mean()}")
