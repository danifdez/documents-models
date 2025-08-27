import numpy as np
import spacy
import networkx as nx
import re
from utils.job_registry import job_handler


@job_handler("key_points")
def key_points(payload) -> dict:
    document = payload.get("document", "")
    key_points_data = _textrank_summarize(document)

    return {"status": "success", "data": key_points_data}


def _textrank_summarize(text, num_sentences=3):
    """
    Simplified implementation for extractive summarization.
    Selects the most important sentences.
    """

    nlp = spacy.load("en_core_web_sm")

    clean_content = re.sub(r"<[^>]+>", "", text)

    sentences = [sent.text for sent in nlp(clean_content).sents]
    sentences = [s for s in sentences if len(s.split()) > 5]

    if not sentences:
        return []

    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            vec1 = nlp(sentences[i]).vector
            vec2 = nlp(sentences[j]).vector
            if vec1.ndim > 0 and vec2.ndim > 0 and np.linalg.norm(vec1) != 0 and np.linalg.norm(vec2) != 0:
                similarity_matrix[i][j] = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            else:
                similarity_matrix[i][j] = 0

    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    final_summary_sentences = []
    seen_sentences = set()
    for score, sentence in ranked_sentences:
        normalized_sentence = re.sub(r'[^\w\s]', '', sentence).lower().strip()
        if normalized_sentence not in seen_sentences:
            final_summary_sentences.append(sentence)
            seen_sentences.add(normalized_sentence)
        if len(final_summary_sentences) >= num_sentences:
            break
    return {"response": final_summary_sentences}