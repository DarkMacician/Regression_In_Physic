from datascience import *
import numpy as np
import matplotlib.pyplot as plt


def sample_proportions_from_scratch(sample_size, list_probabilities, categories=None):
    if not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer.")

    if not isinstance(list_probabilities, list) or not all(isinstance(p, (float, int)) for p in list_probabilities):
        raise TypeError("list_probabilities must be a list of numbers (float or int).")

    total_prob = sum(list_probabilities)
    if not np.isclose(total_prob, 1.0):
        raise ValueError(f"Sum of list_probabilities must be 1.0. Currently, it is {total_prob}.")

    if not all(0 <= p <= 1 for p in list_probabilities):
        raise ValueError("All probabilities must be between 0 and 1.")

    if categories is None:
        categories = list(range(len(list_probabilities)))
    else:
        if len(categories) != len(list_probabilities):
            raise ValueError("Length of categories must match length of list_probabilities.")

    sampled_data = np.random.choice(categories, size=sample_size, p=list_probabilities)

    unique, counts = np.unique(sampled_data, return_counts=True)
    count_dict = dict(zip(unique, counts))

    proportions = {category: count_dict.get(category, 0) / sample_size for category in categories}

    return proportions

sample_size = 1000000000
list_probabilities = [0.25, 0.5, 0.25]

print(sample_proportions_from_scratch(sample_size, list_probabilities))
print(sample_proportions(sample_size, list_probabilities))