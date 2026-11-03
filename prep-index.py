import bm25s
import Stemmer
import pandas as pd
from bm25s.tokenization import Tokenizer

df = pd.read_csv("data/all_ratings811.csv")
corpus = df.chunks.to_list()

stemmer = Stemmer.Stemmer("english")
corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

retriever = bm25s.BM25()
retriever.index(corpus_tokens)
retriever.save("taylor_index_bm25_113", corpus=corpus)