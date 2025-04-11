""""""
from google import genai

import spacy
from spacy.tokens.token import Token
from spacy.tokens.span import Span
from spacy.tokens.doc import Doc

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import nltk
from nltk.collocations import (BigramAssocMeasures, BigramCollocationFinder,
                               TrigramAssocMeasures, TrigramCollocationFinder)

from rake_nltk import Rake

from dotenv import load_dotenv
import os
from functools import partial
from tqdm import tqdm
from pathlib import Path
from typing import Sequence, Optional, Callable
from typeguard import typechecked
from functools import reduce, partial
from collections import defaultdict, Counter


# Global variables
nlp = spacy.load("fr_core_news_sm")


# Personal env variables
load_dotenv()
api_key = os.getenv("API_KEY")
client = genai.Client(api_key=api_key)


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


def titles_from_folder(folder: Path = Path("lotr_corpus")) -> list[str]:
    """Read titles from folder for frontend selection"""
    if not folder.is_dir():
        raise TypeError("Must be a folder")
    documents_path = folder.glob('*.txt')
    titles = [document_path.stem for document_path in documents_path]
    return titles


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
    """"""
    clean_batches = []
    for batch in batches:
        clean_batch = [token for token in batch if not token.is_punct]
        clean_batches.append(clean_batch)
    return clean_batches


def filter_oov(batches: list[list[Token]]) -> list[list[Token]]:
    """"""
    clean_batches = []
    for batch in batches:
        clean_batch = [token for token in batch if not token.is_oov]
        clean_batches.append(clean_batch)
    return clean_batches


def filter_oov_keep_named_entites(batches: list[list[Token]]) -> list[list[Token]]:
    """"""
    clean_batches = []
    for batch in batches:
        clean_batch = [token for token in batch if (not token.is_oov) or token.ent_type_]
        clean_batches.append(clean_batch)
    return clean_batches


def filter_digits(batches: list[list[Token]]) -> list[list[Token]]:
    """"""
    clean_batches = []
    for batch in batches:
        clean_batch = [token for token in batch if not token.is_digit]
        clean_batches.append(clean_batch)
    return clean_batches


def filter_by_pos(batches: list[list[Token]], unwanted_pos_list=None) -> list[list[Token]]:
    """"""
    if not unwanted_pos_list:
        raise ValueError("Pos list not provided")

    clean_batches = []
    for batch in batches:
        clean_batch = [token for token in batch if token.pos_ not in unwanted_pos_list]
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


def tfidf_mean_by_document(
        texts: list[str],
        tokenizer: Callable,
        ngram_range: tuple[int, int] = (1, 1),
        sort=True,
        reverse=True
) -> dict[str, float] | list[tuple[str, float]]:
    """A wrapper function around TfidfVectorizer class in sklearn"""
    # Vectorize text
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=ngram_range)
    X = tfidf_vectorizer.fit_transform(texts)

    # Compute each word's average tf_idf value
    vocab = tfidf_vectorizer.get_feature_names_out()
    X_mean = np.sum(X.toarray(), axis=0) / len(X.toarray())
    word_tfidf_dict = {word: value for word, value in zip(vocab, X_mean)}

    if sort:
        return sorted(word_tfidf_dict.items(), key=lambda x: x[1], reverse=reverse)

    return word_tfidf_dict


def calculate_pmi(texts: list[str], tokenizer: Callable, ngram=2):
    # Combine documents
    text_all = " ".join(texts)
    # Tokenize
    tokens = tokenizer(text_all)
    # Ngram selection
    if ngram == 2:
        finder = BigramCollocationFinder.from_words(tokens)
        measures = BigramAssocMeasures()
    elif ngram == 3:
        finder = TrigramCollocationFinder.from_words(tokens)
        measures = TrigramAssocMeasures()
    else:
        raise ValueError(f"Ngram option can only be 2 or 3. Current value: ngram={ngram}")

    return finder.score_ngrams(measures.pmi)


def calculate_rake(texts: list[str], tokenizer: Callable, max_ngram: int) -> list[tuple[str, float]]:
    r = Rake(language='french', max_length=max_ngram, word_tokenizer=tokenizer)
    r.extract_keywords_from_sentences(texts)
    results = r.get_ranked_phrases_with_scores()
    return [(word, score) for score, word in results]


def metric2concepts(metrics: list[tuple[str, float]], vocab_size: int) -> set[str]:
    """
    :param metrics: a list of (word, score). List should be sorted in advance
    :param vocab_size: only select words with higher scores
    :return: a set of concepts
    """
    return set([example[0] for example in metrics][:vocab_size])


def show_concepts(text: str, concepts: set[str], model=nlp) -> str:
    """
    :param text: "Hello python"
    :param concepts: ["python"]
    :return: "Hello ***python***"
    """
    corpus = model(text)
    return " ".join([f"***{token.text}***" if token.lemma_ in concepts else token.text for token in corpus])


def find_subject(text: str, model=nlp) -> list[int]:
    """
    :param text: text to analyse
    :param model: spacy model
    :return: the list of indices of subjects
    """
    subj_types = ["nsubj", "nsubj:pass"]
    corpus = model(text)
    return [token.i for token in corpus if token.dep_ in subj_types]


def find_sentence_concepts(sent: Span | list[Token],
                           concepts: set[str]
                           ) -> dict[str, list[str]]:
    """
    Find relations on a single sentence
    :param sent: a spacy Span
    :param concepts: set of concepts
    :param model: spacy model
    :return: a dictionary representation of a graph
    """
    subj_types = ["nsubj", "nsubj:pass"]
    subject_concepts = [token.lemma_ for token in sent if token.dep_ in subj_types and token.lemma_.casefold() in concepts]
    related_concepts = [token.lemma_ for token in sent if token.dep_ not in subj_types and token.lemma_.casefold() in concepts]

    if not subject_concepts:
        return dict()
    return {subject_concept: related_concepts for subject_concept in subject_concepts}


def find_concepts_corpus(texts: list[str],
                         concepts: set[str],
                         find_sentence_concepts=find_sentence_concepts
                         ) -> dict[str, list[str]]:
    """
    Similar to find_sentence_concepts but works on an article (multiple sentences)
    :param texts: our corpus
    :param concepts: set of concepts
    :param find_sentence_concepts: function that works on a single sentence
    :return: a dictionary representation of concepts-graph
    """
    corpus = "\n".join(texts)
    pipe = build_pipe(tokenize, batch_by_sentence)
    sentences = pipe(corpus)
    find_concepts = partial(find_sentence_concepts, concepts=concepts)
    relations_list = list(map(find_concepts, sentences))

    relations_dict = defaultdict(list)
    for relations in relations_list:
        for subject_concept, related_concepts in relations.items():
            relations_dict[subject_concept].extend(related_concepts)
    return dict(relations_dict)


def ponderate_relations(relations_dict: dict[str, list[str]]) -> dict[str, Counter[str]]:
    return {subject: Counter(related_concepts) for subject, related_concepts in relations_dict.items()}


def find_frequent_verbes(texts: list[str], model=nlp) -> Counter[str]:
    """Find most frequent verbs"""
    corpus = "\n".join(texts)
    doc = model(corpus)
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    return Counter(verbs)


def generate_prompt(examples, text) -> str:
    prompt = f"""
Vous êtes un ingénieur taliste spécialisé dans la création d'ontologie. Votre mission est de créer une ontologie à partir du roman Le Seigneur des anneaux. Inspirez-vous des exemples de relations donnés ci-dessous, et créez dix à vingt meta-relations qui existent dans le text.

Exemples de relations: {examples}

Texte: {text}

Ne produisez que le résultat dans le format 'tuer | parent_de | habiter' et ne dites rien de plus. Chaque relation doit être séparée par un ' | ' pour faciliter de les parser.
"""
    return prompt


def call_model(contents: str, model: str, client=client):
    response = client.models.generate_content(
        model=model,
        contents=contents,
    )
    return response
