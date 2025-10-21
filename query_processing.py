import numpy as np
from typing import List, Dict
from mistralai import Mistral
from similarity_search import search_similar_chunks

def retrieve_context(query: str, embeddings: np.ndarray, metadata: List[Dict], mistral_client: Mistral, top_k: int = 10) -> List[Dict]:
    '''Retrieve most relevant chunks'''
    # Get query embedding
    response = mistral_client.embeddings.create(model='mistral-embed', inputs=[query])
    query_embedding = np.array(response.data[0].embedding)
    
    # Search for similar chunks
    return search_similar_chunks(query_embedding, embeddings, metadata, top_k)