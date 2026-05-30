# Fungsi: Mencari user paling mirip dengan user input.

import argparse
import json
import numpy as np
from compute_scores import pearson_score


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find users who are similar to the input user')
    parser.add_argument('--user', dest='user', required=True, help='Input user')
    return parser


def find_similar_users(dataset, user, num_users):
    if user not in dataset:
        raise TypeError('User not found in dataset')
    scores = np.array([[x, pearson_score(dataset, user, x)] for x in dataset if x != user])
    scores_sorted = np.argsort(scores[:, 1])[::-1]
    top_users = scores_sorted[:num_users]
    return scores[top_users]


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    with open('ratings.json', 'r') as f:
        data = json.loads(f.read())

    print(f'\nUsers similar to {args.user}:\n')
    similar_users = find_similar_users(data, args.user, 3)
    print('User\t\t\tSimilarity score')
    print('-' * 41)
    for item in similar_users:
        print(f"{item[0]}\t\t{round(float(item[1]), 2)}")