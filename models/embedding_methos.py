from sentence_transformers import SentenceTransformer
from typing import List, Union

# Preload models to avoid reloading on each request
MODELS = {
    "MiniLM": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    "DistilBERT": SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")
}

def generate_embeddings(
    docs: Union[str, List[str], List[object]],
    model_name: str = "MiniLM"
):
    """
    Generate embeddings using the selected model.
    Supports single string, list of strings, or list of LangChain Documents.
    """
    if model_name not in MODELS:
        raise ValueError(f"Invalid model. Choose from: {list(MODELS.keys())}")

    model = MODELS[model_name]

    if isinstance(docs, str):
        texts = [docs]
    elif isinstance(docs, list) and isinstance(docs[0], str):
        texts = docs
    else:
        texts = [doc.page_content for doc in docs]

    embeddings = model.encode(texts, convert_to_tensor=False)

   
    if isinstance(docs, str):
        return embeddings[0]  # returns a single 1D vector

    return embeddings  # returns list of vectors
