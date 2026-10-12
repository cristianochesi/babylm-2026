"""Shared helpers for token-budget training (drop next to the trainers)."""

def make_babylm_save_points(max_millions=None):
    """Required BabyLM checkpoint schedule, in MILLIONS of words/tokens:
    every 1M to 10M, every 10M to 100M, every 100M to 1000M.
    Optionally truncate at max_millions."""
    pts = list(range(1, 10))                 # 1..9
    pts += list(range(10, 100, 10))          # 10,20,..,90
    pts += list(range(100, 1001, 100))       # 100,200,..,1000
    if max_millions is not None:
        pts = [p for p in pts if p <= max_millions]
        if max_millions not in pts:
            pts.append(int(max_millions))
    return sorted(set(pts))


def epoch_iter(loader, epochs):
    """Yield every batch of `loader`, `epochs` times, then stop (no cycling)."""
    for _ in range(max(1, epochs)):
        for batch in loader:
            yield batch
