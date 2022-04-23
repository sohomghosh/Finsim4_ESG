from glob import glob
import json
import numpy as np
import os
import pdftotext
from nltk import word_tokenize
from nltk.corpus import stopwords

STOP = set(stopwords.words("english"))


def get_pdf_paths(base_path="data/English_reports"):
    paths = glob(base_path + "/**/*.pdf", recursive=True)
    return list(paths)


def get_pdf_text(path):
    with open(path, "rb") as f:
        pdf = pdftotext.PDF(f)
    text = "\n\n".join(pdf)
    return text


def count_tokens(path="data/English_reports"):
    count = 0
    for f in os.listdir(path):
        text = get_pdf_text(os.path.join(path, f))
        tokens = word_tokenize(text)
        print(len(tokens))
        count += len(tokens)

    print(f"There are {count} tokens in the corpus")


def read_keywords(path="data/terms/ESG_taxonomy_train.json"):
    f = json.load(open(path))
    data = []
    for l in f:
        data.append(l["term"])
    return data


def load_glove_model(gloveFile):
    # https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
    print("Loading Glove Model")
    f = open(gloveFile, "r")
    model = {}
    for line in f:
        split_line = line.split()
        if len(split_line) > 100:
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
    print("Done.", len(model), " word vectors loaded!")
    return model


def text_to_vector(s, model, k=100):
    s = s.lower()
    l_s = word_tokenize(s)
    l_s_found = [a for a in l_s if (a in model and a and a not in STOP)]
    if l_s_found:
        out = 0
        for a in l_s_found:
            out += model[a]

        out = out / len(l_s_found)

        out = out / (np.linalg.norm(out) + 1e-9)

        return out
    else:
        return np.zeros((k,))
