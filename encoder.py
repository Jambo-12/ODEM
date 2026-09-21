import os

import numpy as np
from openai import OpenAI


SUPPORTED_MODEL = "text-embedding-3-large"


def encode_sentences(sentence_list, model_name):
    """Encode one batch with the experiment's unchanged embedding model."""
    if model_name != SUPPORTED_MODEL:
        raise ValueError("Unsupported text embedding model: {}".format(model_name))
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing text embedding API key; set OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").strip()
    client = OpenAI(api_key=api_key, base_url=api_base)
    response = client.embeddings.create(input=sentence_list, model=model_name)
    embeddings = [
        np.asarray(item.embedding).reshape(1, -1)
        for item in response.data
    ]
    return np.concatenate(embeddings, axis=0)
