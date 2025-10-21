import numpy as np
import json
from typing import List, Dict, Tuple
from pathlib import Path

def load_embeddings_from_files(embeddings_dir: str = './embeddings') -> Tuple[np.ndarray, List[Dict]]:
    '''Load embeddings and metadata from files'''
    embeddings_path = Path(embeddings_dir)
    embeddings = np.load(embeddings_path / 'embeddings.npy')
    with open(embeddings_path / 'metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return embeddings, metadata

def search_similar_chunks(query_embedding: np.ndarray, embeddings: np.ndarray, metadata: List[Dict], top_k: int = 10) -> List[Dict]:
    '''Find top_k most similar chunks using cosine similarity'''
    # Calculate cosine similarities
    similarities = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))
    
    # Get top_k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return results with scores
    results = []
    for idx in top_indices:
        result = metadata[idx].copy()
        result['similarity_score'] = float(similarities[idx])
        results.append(result)
    
    return results


# Removed all the complex functions - keeping only the essentials