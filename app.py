import streamlit as st
import bm25s
import utils as U
import Stemmer
from datetime import datetime
import numpy as np

st.title("Press Review Search")

retriever = U.load_index()
df = U.load_data()
stemmer = Stemmer.Stemmer("english")

query = st.text_input("Input a query")
if not query.strip():
    st.warning("Please input a query to get started.")
else:
    terms = U.query_terms(query)
    query_tokens = bm25s.tokenize(query, stemmer=stemmer)
    num_results = st.number_input("Maximum results to see", value=1000)
    num_results_per_page = st.number_input("How many results to see per page", value=10)

    results, scores = retriever.retrieve(query_tokens, k=num_results)
    scaled_scores = (scores - np.min(scores))/(np.max(scores) - np.min(scores))
    actual_num_results = np.count_nonzero(scaled_scores)
    total_pages = actual_num_results//num_results_per_page

    with st.container():
        page = st.pagination(num_pages=int(total_pages) if total_pages > 1 else 1)

    start_idx = (page - 1) * num_results_per_page
    end_idx = start_idx + num_results_per_page

    st.write(f"Found **{actual_num_results}** results.")
    for i in range(num_results_per_page):
        doc, score = results[0, start_idx+i], scaled_scores[0, start_idx+i]
        if score > 0:
            row_data = U.lookup_by_index(df, doc["id"])
            title = row_data['Title']
            author = row_data["Author"]
            date = datetime.strptime(row_data["Date"], "%Y-%m-%d")
            newspaper = row_data["Newspaper"]
            source = row_data["Source"]
            np_type = row_data["newspaper_type"]
            text = row_data["chunks"]
            with st.expander(f"{title} (Relevance score: {score :.3f})"):
                st.write(f"By {author} on {date.strftime("%B %d, %Y")} from **{newspaper}**")
                st.html(f"<p>{U.highlight(text, terms)}</p>")