import click
import json
import logging
import numpy
import random
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from gensim.models import KeyedVectors
from utils.scorer import compute_accuracy, compute_average_rank
logging.basicConfig(level="INFO")


def get_X_Y(embeddings, term_list):
    X, Y = [], []
    for term in term_list:
        phrase_vector = numpy.zeros((embeddings.vector_size,))
        for word in word_tokenize(term["term"]):
            try:
                phrase_vector += embeddings[word.lower()]
            except KeyError:
                logging.debug(f"Word {word.lower()} not found.")
                pass
        if type(X) == list:
            X = numpy.array(phrase_vector)
        else:
            X = numpy.vstack([X, phrase_vector])
        Y.append(term["concept"])
    return X, Y


def train(X, Y):
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X, Y)
    return model


@click.command()
@click.option('--embeddings_path', '-e', help='path to Gensim embedding model', default="models/ESG_reports_custom_w2v_300d.txt")
@click.option('--terms_path', '-t', help='path to terms dataset', default="data/terms/ESG_taxonomy_train.json")
def main(embeddings_path, terms_path):
    term_list = json.load(open(terms_path))
    embeddings = KeyedVectors.load_word2vec_format(embeddings_path)

    # Train
    random.shuffle(term_list)
    train_list = term_list[:-int(0.8*len(term_list))]
    test_list = term_list[-int(0.8*len(term_list)):]
    X_train, Y_train = get_X_Y(embeddings, train_list)
    model = train(X_train, Y_train)

    # Test
    predictions = []
    X_test, Y_test = get_X_Y(embeddings, test_list)
    probas = model.predict_proba(X_test).tolist()

    [predictions.append([x[1] for x in [sorted(zip(example, model.classes_), reverse=True)][0]]) for example in probas]

    # Score
    accuracy = compute_accuracy(Y_test, predictions)
    average_rank = compute_average_rank(Y_test, predictions)
    logging.info(f"Average Rank : {average_rank}")
    logging.info(f"Accuracy : {accuracy}")

    # Dump data
    data = []
    for idx, example in enumerate(predictions):
        data.append({"term": test_list[idx]["term"], "concept": test_list[idx]["concept"], "predicted_concepts": example})
    json.dump(data, open("data/outputs/baseline_2_ESG_taxonomy_predictions_300.json", "w"), indent=4)


if __name__ == "__main__":
    main()
