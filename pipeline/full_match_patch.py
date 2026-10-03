"""
Pass 2 of the chapter nav wiring: once every chapter AND the match-index
page have been published once (so their real artifact URLs exist), patch
each chapter's already-built HTML file in place, replacing the three nav
placeholder tokens with real URLs. Cheap string replace on disk -- no need
to re-run the head/anim/heat/tail concatenation.

Usage: python3 full_match_patch.py urls.json
  urls.json: {"chapters": {"1": "https://...", "2": "https://...", ...},
              "index": "https://..."}
"""
import json
import sys

from config import CHAPTERS_DIR


def main(urls_path):
    with open(urls_path) as f:
        urls = json.load(f)
    chapter_urls = urls['chapters']
    index_url = urls['index']

    with open(f'{CHAPTERS_DIR}/manifest.json') as f:
        manifest = json.load(f)
    chapters = manifest['chapters']

    for ch in chapters:
        idx = ch['chapterIndex']
        path = ch['htmlPath']
        with open(path, 'r', encoding='utf-8') as f:
            html = f.read()

        prev_url = chapter_urls.get(str(idx - 1)) if idx > 1 else None
        next_url = chapter_urls.get(str(idx + 1)) if idx < len(chapters) else None

        if prev_url:
            html = html.replace('__PREV_URL__', prev_url)
        if next_url:
            html = html.replace('__NEXT_URL__', next_url)
        html = html.replace('__INDEX_URL__', index_url)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"patched chapter {idx:02d}: {path}")

    print("Done. Republish each chapters/pressure_read_XX.html with its own url= to go live.")


if __name__ == '__main__':
    main(sys.argv[1])
