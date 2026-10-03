"""
Merge classification.py's output (match_features_classified.pkl: per-frame
depth/inter-line/structure/phase) with pressure.py's output (match_pressure.pkl:
per-frame pressure_score/pressing_team_id/primary_presser_jersey) into the single
table animation_export.py reads.

This step didn't previously exist as a saved script -- match_features_with_pressure.pkl
had been produced by hand in an earlier session, and that ad hoc merge silently
dropped primary_presser_jersey/primary_presser_contribution. The effect: every
frame's `primaryPresser` in the exported animation came out None regardless of
pressure_score, which read in the artifact as "88% pressure, but no single
defender applying sustained pressure" -- a real bug, not the documented (and
legitimate) case where pressure is genuinely spread across multiple defenders.
Found while re-running the pipeline for the classification threshold change;
fixed here so it's a real, re-runnable step instead of a one-off.

Left join on frameNum keeps every classified row (including the ~31% with no
resolved pressure -- no possession, or possession but no defender close enough
to matter -- which correctly get null pressure fields, not a merge artifact).
match_pressure.pkl is deduped on frameNum first as a defensive measure against
this match's known duplicate-frameNum quirk (58 frameNum values repeat up to
16x in the raw tracking file -- see literature-and-data-foundations project
notes) -- classification.py's own source data is assumed already deduped
upstream of it, so only the second input needs the guard here.
"""
import pandas as pd
from config import pkl_path

CLASSIFIED_PATH = pkl_path('match_features_classified')
PRESSURE_PATH = pkl_path('match_pressure')
OUT_PATH = pkl_path('match_features_with_pressure')

PRESSURE_COLS = ['frameNum', 'pressure_score', 'pressing_team_id',
                  'primary_presser_jersey', 'primary_presser_contribution']


def merge_pressure(classified_path=CLASSIFIED_PATH, pressure_path=PRESSURE_PATH):
    mc = pd.read_pickle(classified_path)
    mp = pd.read_pickle(pressure_path).drop_duplicates(subset='frameNum', keep='first')
    merged = mc.merge(mp[PRESSURE_COLS], on='frameNum', how='left')
    assert len(merged) == len(mc), \
        f"merge changed row count: {len(mc)} -> {len(merged)} (duplicate frameNum in pressure input?)"
    return merged


if __name__ == '__main__':
    merged = merge_pressure()
    print(f"Merged {len(merged)} rows.")
    print("primary_presser_jersey populated:",
          merged['primary_presser_jersey'].notna().sum(), '/', len(merged))
    merged.to_pickle(OUT_PATH)
    print(f"Saved {OUT_PATH}")
