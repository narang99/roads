from typing import List, Any

def chunk_list(data: List[Any], n_chunks: int) -> List[List[Any]]:
    """
    Split a list into n_chunks as evenly as possible.

    Example:
    data = [1, 2, 3, 4, 5]
    n_chunks = 3
    -> [[1, 2], [3, 4], [5]]
    """
    if n_chunks <= 0:
        raise ValueError("n_chunks must be > 0")

    n = len(data)
    k, r = divmod(n, n_chunks)

    chunks = []
    start = 0
    for i in range(n_chunks):
        size = k + (1 if i < r else 0)
        chunks.append(data[start:start + size])
        start += size

    return chunks