from pathlib import Path
import pickle


def load_pickle(path: str):
    """Load any pickled Python object from *path*."""
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def save_pickle(obj, path: str):
    """Save *obj* to *path*; create parent directories if absent."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump(obj, fh)

