""""""
import streamlit as st

from utils import *

import inspect
import os
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Sequence, Optional
from functools import partial


name2func = {
    # Batch options
    "article":  batch_by_article,
    "sentence": batch_by_sentence,
    "token": batch_by_token,
    # Filter options
    "stopwords": filter_stopwords,
    "punctuations": filter_punctuations,
    "digits": filter_digits,
    "oov": filter_oov,
    # Output options
    "str": to_str,
    "str_lower": to_str_lower,
    "lemma": to_lemma,
    # Squeeze
    "squeeze": squeeze,
}


if "corpus" not in st.session_state:
    st.session_state.corpus = texts_from_folder(Path("./lotr_corpus"))


st.title("Build pipeline")

with st.form("Tokenizer options"):
    # Tokenizer options
    batch = st.selectbox("Batch option:", options=["article", "sentence", "token"])
    filters = st.multiselect("Filter options (multiple)", options=["stopwords", "punctuations", "digits", "oov_keep_named_entities"])
    output = st.selectbox("Output mode:", options=["str", "str_lower", "lemma"])
    squeeze_name = st.selectbox("Squeeze function", options=["squeeze"])
    # Mesure
    mesure = st.selectbox("Mesure:", options=["tfidf", "pmi", "Rake"])
    # Submit button
    tokenizer_button = st.form_submit_button("Build pipe")


if all([batch, filters, output, squeeze_name, tokenizer_button]):
    # Start analysing corpus if all options provided
    # Build tokenizer
    batch_func = name2func[batch]
    filter_func = build_pipe([[name2func[filt] for filt in filters]])
    output_func = name2func[output]
    squeeze_func = name2func[squeeze_name]
    pipe = build_pipe(batch_func, filter_func, output_func, squeeze_func)


if mesure:
    st.title("Customize mesure")

    if mesure == "tfidf":
        with st.form("tfidf customization"):
            st.write("tfidf")
    elif mesure == "pmi":
        with st.form("pmi customization"):
            st.write("pmi")
    elif mesure == "Rake":
        with st.form("Rake customization"):
            st.write("Rake")

