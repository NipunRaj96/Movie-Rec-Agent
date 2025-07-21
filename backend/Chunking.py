from typing import List

def chunk_text(text: str, max_length: int = 512) -> List[str]:
    """
    Chunk text into smaller segments for processing.
    
    Args:
        text: Input text to chunk
        max_length: Maximum length per chunk in words
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Split by sentences first, then by words if needed
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) <= max_length:
            current_chunk.extend(words)
            current_length += len(words)
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = words
            current_length = len(words)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks 