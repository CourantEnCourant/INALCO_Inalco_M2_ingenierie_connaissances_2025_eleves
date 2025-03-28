""""""
import spacy
from spacy.tokens.token import Token
from spacy.tokens.doc import Doc

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Sequence, Optional, Callable
from typeguard import typechecked
from functools import reduce


# Global variables
nlp = spacy.load("fr_core_news_sm")


def texts_from_folder(folder: Path = Path("lotr_corpus")) -> list[str]:
    """Generate corpus from folder"""
    if not folder.is_dir():
        raise TypeError("Must be a folder")
    texts = []
    documents_path = folder.glob('*.txt')
    for document_path in documents_path:
        with open(document_path, 'r') as f:
            text = f.read()
            texts.append(text)
    return texts


# A magical higher-order function
def build_pipe(*funcs):
    """Combine functions into one"""
    return lambda x: reduce(lambda acc, f: f(acc), funcs, x)


# Tokenize
def tokenize(text: str, model=nlp) -> Doc:
    return model(text)


# Batch functions
def batch_by_token(spacy_doc: Doc) -> list[list[Token]]:
    return [[token] for token in spacy_doc]


def batch_by_sentence(spacy_doc: Doc) -> list[list[Token]]:
    sentences = []
    for sent in spacy_doc.sents:
        sentence = [token for token in sent]
        sentences.append(sentence)
    return sentences


def batch_by_article(spacy_doc: Doc) -> list[list[Token]]:
    return [[token for token in spacy_doc]]


# Filter functions
def filter_stopwords(batches: list[list[Token]]) -> list[list[Token]]:
    """Remove stopwords"""
    clean_batches = []
    for batch in batches:
        clean_batch = [token for token in batch if not token.is_stop]
        clean_batches.append(clean_batch)
    return clean_batches


def filter_punctuations(batches: list[list[Token]]) -> list[list[Token]]:
    """Remove stopwords"""
    clean_batches = []
    for batch in batches:
        clean_batch = [token for token in batch if not token.is_punct]
        clean_batches.append(clean_batch)
    return clean_batches


# Output mode functions
def to_str(batches: list[list[Token]]) -> list[list[str]]:
    str_batches = []
    for batch in batches:
        str_batch = [token.text for token in batch]
        str_batches.append(str_batch)
    return str_batches


def to_str_lower(batches: list[list[Token]]) -> list[list[str]]:
    str_batches = []
    for batch in batches:
        str_batch = [token.text.lower() for token in batch]
        str_batches.append(str_batch)
    return str_batches


def to_lemma(batches: list[list[Token]]) -> list[list[str]]:
    str_batches = []
    for batch in batches:
        str_batch = [token.lemma_ for token in batch]
        str_batches.append(str_batch)
    return str_batches


def squeeze(multi_dim_list: list) -> list:
    """Only squeeze the first axis: e.g. (1, 3) -> (3,)"""
    if len(multi_dim_list) == 1:
        return multi_dim_list[0]
    raise Exception(f"挤不出来. First axis size is: {len(multi_dim_list)}")


def tf_idf_mean_by_document(texts: list[str], tokenizer: Callable, n_gram: int = 1) -> dict[str, float]:
    """A wrapper function around TfidfVectorizer class in sklearn"""
    # Vectorize text
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(n_gram, n_gram))
    X = tfidf_vectorizer.fit_transform(texts)

    # Compute each word's average tf_idf value
    words = tfidf_vectorizer.get_feature_names_out()
    X_arr = X.toarray()
    X_mean = np.sum(X_arr, axis=0) / len(X_arr)

    return {word: value for word, value in zip(words, X_mean)}
