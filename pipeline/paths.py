"""Where the project, the raw match data and the generated outputs live.

Defaults work straight from a clone of the repository:

    <repo>/data/<match id>/   raw match files (metadata.json, roster.json, events.json, tracking.jsonl.bz2)
    <repo>/output/            everything the pipeline generates (feature tables, chapters, images, site)

Override any of them with environment variables:

    DSM_ROOT   project root (default: the folder above pipeline/)
    DSM_DATA   data folder   (default: <root>/data)
    DSM_OUT    output folder (default: <root>/output)
"""
import os

ROOT = os.path.abspath(os.environ.get('DSM_ROOT') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA = os.path.abspath(os.environ.get('DSM_DATA') or os.path.join(ROOT, 'data'))
OUT = os.path.abspath(os.environ.get('DSM_OUT') or os.path.join(ROOT, 'output'))
TEMPLATES = os.path.join(ROOT, 'templates')


def root(*p):
    return os.path.join(ROOT, *p)


def out(*p):
    """A path in the output folder; parent folders are created on demand."""
    path = os.path.join(OUT, *p)
    os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path, exist_ok=True)
    return path


def data_dir(raw_dir):
    """Resolve a registry raw_dir: absolute paths are used as given; 'data/<id>' is placed under DATA."""
    if os.path.isabs(raw_dir):
        return raw_dir
    rel = raw_dir[5:] if raw_dir.startswith('data/') else raw_dir
    return os.path.join(DATA, rel)
