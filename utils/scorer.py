import argparse
import json
import logging
import numpy as np
from sklearn import metrics


def get_rank(gold, list_predictions, max_k=3):
    list_predictions = list_predictions[:max_k]
    try:
        rank = list_predictions.index(gold) + 1
    except ValueError:
        rank = max_k + 1
    return rank


def compute_average_rank(Y_test, predictions):
    ranks = []
    for idx, y in enumerate(Y_test):
        logging.debug(f"y={Y_test[idx]}, predictions={predictions[idx]}")
        ranks.append(get_rank(Y_test[idx], predictions[idx]))
    return np.mean(ranks)


def compute_accuracy(Y_test, predictions):
    predictions = [prediction[0] for prediction in predictions]
    return metrics.accuracy_score(Y_test, predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--predicted_data_path', default="data/outputs/baseline_1_predictions.json")
    parser.add_argument('--gold_data_path', default="data/outputs/baseline_1_predictions.json")
    parser.add_argument('--max_k', default=3, type=int)

    args = parser.parse_args()

    max_k = args.max_k

    data = json.load(open(args.predicted_data_path, "r"))
    gold = json.load(open(args.predicted_data_path, "r"))

    Y_test = [entity["concept"] for entity in gold]
    predictions = [entity["predicted_concepts"] for entity in data]

    average_rank = compute_average_rank(Y_test, predictions)
    accuracy = compute_accuracy(Y_test, predictions)

    print(f"Average Rank : {average_rank}")
    print(f"Accuracy : {accuracy}")
