"""Keep new experiment outputs separate from recorded evidence."""
from pathlib import Path


def guard_output(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    for name in ['results', 'data', 'figures', 'archive', 'docs', 'src', 'tests', 'scripts', 'notebooks']:
        protected = root / name
        if path == protected or protected in path.parents:
            raise ValueError(f'Recorded/project files are protected: {path}. Use a fresh runs/ directory.')
    if path == root:
        raise ValueError('Choose a separate output directory, such as runs/sird.')
    return path
