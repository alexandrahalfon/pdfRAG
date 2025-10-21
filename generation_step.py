from typing import List, Dict
import numpy as np
from mistralai import Mistral
from query_processing import retrieve_context

def generate_answer_with_context(query: str, retrieved_chunks: List[Dict], mistral_client: Mistral) -> str:
    '''Generate response using retrieved chunks as context'''
    if not retrieved_chunks or retrieved_chunks[0]['similarity_score'] < 0.3:
        return "I couldn't find relevant information in the documents to answer your question."
    
    # Build context from retrieved chunks
    context = '\n\n'.join([f"Page {chunk['page']}: {chunk['text']}" for chunk in retrieved_chunks])
    
    # Generate response
    response = mistral_client.chat.complete(
        model='mistral-large-latest',
        messages=[
            {'role': 'system', 'content': 'Answer questions based only on the provided document context. Be concise and accurate.'},
            {'role': 'user', 'content': f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ],
        temperature=0.1,
        max_tokens=600
    )
    
    return response.choices[0].message.content


def ask_document(query: str, embeddings: np.ndarray, metadata: List[Dict], mistral_client: Mistral, top_k: int = 5, show_sources: bool = True) -> Dict:
    '''Ask question about document'''
    # Simple greeting check
    if query.lower().strip() in ['hello', 'hi', 'hey']:
        return {'query': query, 'answer': "Hello! Ask me questions about your uploaded documents.", 'sources': []}
    
    # Retrieve relevant context and generate answer
    retrieved_chunks = retrieve_context(query, embeddings, metadata, mistral_client, top_k)
    answer = generate_answer_with_context(query, retrieved_chunks, mistral_client)
    
    return {'query': query, 'answer': answer, 'sources': retrieved_chunks}