# Fungsi: Menghitung Similarity Score (Euclidean / Pearson) antar dua user.

import argparse
import json
import numpy as np


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--user1', dest='user1', required=True, help='First user')
    parser.add_argument('--user2', dest='user2', required=True, help='Second user')
    parser.add_argument("--score-type", dest="score_type", required=True, choices=['Euclidean', 'Pearson'],
                        help='Similarity metric')
    return parser


def euclidean_score(dataset, user1, user2):
    if user1 not in dataset or user2 not in dataset:
        raise TypeError('User not found in dataset')
    common_movies = {item: 1 for item in dataset[user1] if item in dataset[user2]}
    if len(common_movies) == 0: return 0
    squared_diff = [np.square(dataset[user1][item] - dataset[user2][item]) for item in common_movies]
    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


def pearson_score(dataset, user1, user2):
    if user1 not in dataset or user2 not in dataset:
        raise TypeError('User not found in dataset')
    common_movies = {item: 1 for item in dataset[user1] if item in dataset[user2]}
    num_ratings = len(common_movies)
    if num_ratings == 0: return 0

    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])
    user1_sq = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_sq = np.sum([np.square(dataset[user2][item]) for item in common_movies])
    sum_prod = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])

    Sxy = sum_prod - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_sq - np.square(user1_sum) / num_ratings
    Syy = user2_sq - np.square(user2_sum) / num_ratings
    if Sxx * Syy == 0: return 0
    return Sxy / np.sqrt(Sxx * Syy)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    with open('ratings.json', 'r') as f:
        data = json.loads(f.read())

    score = euclidean_score(data, args.user1, args.user2) if args.score_type == 'Euclidean' else pearson_score(data,
                                                                                                               args.user1,
                                                                                                               args.user2)
    print(f"\n{args.score_type} score: {score}")