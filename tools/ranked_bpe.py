"""Exact lowest-rank BPE with bounded short-piece reuse."""
from functools import lru_cache
from heapq import heappop, heappush


def merge_ranked(word, ranks):
    # Short words are cheaper with the original bounded scan. Long pieces use
    # adjacent-pair updates instead of rescanning the entire list after a merge.
    parts = list(word)
    if len(parts) <= 32:
        while len(parts) > 1:
            best, rank = None, None
            for i in range(len(parts) - 1):
                candidate = ranks.get((parts[i], parts[i + 1]))
                if candidate is not None and (rank is None or candidate < rank):
                    best, rank = i, candidate
            if best is None:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        return parts
    count = len(parts)
    previous = list(range(-1, count - 1))
    following = list(range(1, count)) + [-1]
    version = [0] * count
    alive = [True] * count
    heap = []

    def offer(left):
        if left < 0:
            return
        right = following[left]
        if right < 0:
            return
        rank = ranks.get((parts[left], parts[right]))
        if rank is not None:
            # Original positions retain the reference's leftmost tie breaking.
            heappush(heap, (rank, left, right, version[left], version[right]))

    for i in range(count - 1):
        offer(i)
    while heap:
        _, left, right, lv, rv = heappop(heap)
        if (not alive[left] or not alive[right] or following[left] != right
                or version[left] != lv or version[right] != rv):
            continue
        parts[left] += parts[right]
        version[left] += 1
        alive[right] = False
        following[left] = following[right]
        if following[right] >= 0:
            previous[following[right]] = left
        offer(previous[left])
        offer(left)
    return [part for i, part in enumerate(parts) if alive[i]]


def piece_encoder(ranks, ids, byte_map):
    def encode(piece):
        mapped = "".join(byte_map[b] for b in piece.encode("utf-8"))
        return tuple(ids[token] for token in merge_ranked(mapped, ranks))

    cached = lru_cache(maxsize=8192)(encode)

    def bounded(piece):
        # No arbitrary long document or piece is retained. The cache is bound
        # to this vocabulary instance; special-token parsing remains upstream.
        return cached(piece) if len(piece) <= 128 else encode(piece)

    return bounded
