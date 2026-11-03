import streamlit as st
import pandas as pd
import bm25s
import json
import re
import html
import Stemmer

stemmer = Stemmer.Stemmer("english")
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

@st.cache_data
def load_index():
    retriever = bm25s.BM25.load("taylor_index_bm25_113", load_corpus=True)
    return retriever

@st.cache_data
def load_data():
    df = pd.read_csv("data/all_ratings811.csv")
    return df

def lookup_by_index(df, id):
    return json.loads(df.iloc[id].to_json())

def query_terms(query):
    toks = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, return_ids=False, show_progress=False)
    return set(toks[0])

def highlight(text, terms):
    out, last = [], 0
    for m in TOKEN_RE.finditer(text):
        word = m.group()
        if stemmer.stemWord(word.lower()) in terms:
            out.append(html.escape(text[last:m.start()]))
            out.append(f"<mark>{html.escape(word)}</mark>")
            last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)