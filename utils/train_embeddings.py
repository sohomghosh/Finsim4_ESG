import multiprocessing
import logging
import nltk
from gensim.models import Word2Vec
from tqdm import tqdm

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import word_tokenize

from utils.text_processing import get_pdf_paths, get_pdf_text

logging.basicConfig(level="INFO")


def main():
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    pdf_paths = get_pdf_paths()
    sentences = []

    logging.info("Reading files...")

    for i, pdf_path in enumerate(tqdm(pdf_paths)):
        text = get_pdf_text(pdf_path)
        for sentence in sent_detector.tokenize(text.lower().strip()):
            sentences.append(word_tokenize(sentence))

    dim = 100
    cpus = multiprocessing.cpu_count() - 2  # leave 2 CPUs free
    logging.info("Training model with {} CPU workers and {} dimensions...")
    model = Word2Vec(sentences=sentences, sg=1, vector_size=dim, workers=cpus)

    logging.info("Saving model...")
    model.wv.save_word2vec_format(f"models/ESGreports_custom_w2v_{dim}d.txt")


if __name__ == "__main__":
    main()
