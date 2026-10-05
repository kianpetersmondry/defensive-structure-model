"""
Shared config + data loading for the defensive-structure pipeline.

Match-aware as of round 30: every downstream script (tracking_features.py,
transitions.py, classification.py, pressure.py, engagement.py,
compactness_tag.py, events_processing.py, animation_export.py,
full_match_export.py, ...) imports module-level constants from here rather
than a match id, so adding a second match must not change any of their
import statements. Instead, which match's data these constants point at is
selected by the MATCH_ID environment variable at import time -- unset (or
'10508') behaves exactly as before (Morocco vs Spain, Round of 16); setting
MATCH_ID=10515 before running a script points every constant at Morocco vs
France (Semifinal) instead. Run scripts like:
    MATCH_ID=10515 python3 tracking_features.py
Add a new match by adding one entry to MATCHES below -- nothing else in this
file, or in any importing script, needs to change.
"""
import json
import os

# Per-match facts live in pipeline/matches.json (one registry for the whole project); this table is
# generated from it with the same keys every script already uses. Paths come from paths.py.
from registry import R as _R
from paths import ROOT as PW_ROOT, OUT as OUT_ROOT

MATCHES = {
    _m['id']: dict(
        game_dir=_R.raw_dir(_m['id']),
        home_team_id=_m['home']['id'], away_team_id=_m['away']['id'],
        home_team_name=_m['home']['name'], away_team_name=_m['away']['name'],
        home_color=_m['home']['color'], home_color_dim=_m['home']['color_dim'],
        away_color=_m['away']['color'], away_color_dim=_m['away']['color_dim'],
        head_template=_R.head_template(_m['id']),
        label=_m['label'],
        **({'heatmap_vmax_log': _m['heatmap_vmax_log']} if _m.get('heatmap_vmax_log') is not None else {}),
    ) for _m in (_R.get(i) for i in _R.ids())
}

MATCH_ID = os.environ.get('MATCH_ID', '10508')
_match = MATCHES[MATCH_ID]

GAME_DIR = _match['game_dir']
ROSTER_PATH = f'{GAME_DIR}/roster.json'
METADATA_PATH = f'{GAME_DIR}/metadata.json'
EVENTS_PATH = f'{GAME_DIR}/events.json'
TRACKING_PATH = f'{GAME_DIR}/tracking.jsonl.bz2'

HOME_TEAM_ID = _match['home_team_id']
AWAY_TEAM_ID = _match['away_team_id']
HOME_TEAM_NAME = _match['home_team_name']
AWAY_TEAM_NAME = _match['away_team_name']
HOME_COLOR = _match['home_color']
HOME_COLOR_DIM = _match['home_color_dim']
AWAY_COLOR = _match['away_color']
AWAY_COLOR_DIM = _match['away_color_dim']
MATCH_LABEL = _match['label']
GAME_ID = MATCH_ID
HEAD_TEMPLATE = _match['head_template']

# Fixed log-scale brightness ceiling for the full-match danger heatmap
# (full_match_heatmap.py), per match -- round 37. Originally every match
# reused ONE constant (4.375829836329568) carried over from the very first
# round-14 fix, which was only ever calibrated against match 1's small
# hand-picked 5-minute highlight window. Reused unmodified across every
# match's full ~130-minute chapter export, it under-calibrated every match
# (each one's real danger-density distribution runs "hotter" over a full
# match than that one curated window did) -- worst for match 5, whose own
# 99.5th percentile (6.21) is 42% higher than the old shared constant. The
# visible symptom was exactly the pre-round-14 bug re-appearing in specific
# frames: a hard-edged plateau instead of a graded glow. Fixed by giving
# each match its own ceiling, computed the same way (99.5th percentile of
# log1p(mass/LOG_FLOOR) sampled across that match's own full set of
# chapters, not just one window) -- still one fixed number PER MATCH so
# brightness stays comparable chapter-to-chapter within a match, the
# original round-17 design goal, just no longer borrowed from a different
# match's calibration.
def _heat_ceiling():
    """The registry's value; else the one heat_ceiling.py computed for this match; else the old shared constant."""
    if _match.get('heatmap_vmax_log') is not None:
        return _match['heatmap_vmax_log']
    path = os.path.join(OUT_ROOT, 'derived', f'heat_ceiling_{MATCH_ID}.json')
    if os.path.exists(path):
        return json.load(open(path))['heatmap_vmax_log']
    return 4.375829836329568


HEATMAP_VMAX_LOG = _heat_ceiling()

# The Match Library hub -- same URL for every match, so this is a plain
# constant rather than per-match config. Used to make each chapter's own
# masthead title a clickable link back to the hub (round 35).
HUB_URL = 'https://claude.ai/code/artifact/ffac7d5a-5ef4-409d-9586-80b044e97f16'

# Every intermediate pickle/json the pipeline scripts (tracking_features.py ->
# transitions.py -> classification.py -> pressure.py -> merge_pressure.py ->
# engagement.py -> compactness_tag.py -> full_match_export.py -> ...) read or
# write lives directly under project_work as a bare filename. For the
# original match (10508) that filename is unchanged from before this file
# became match-aware, so nothing already published or previously computed for
# it is at risk of being silently overwritten by a second match's run.
# Anything else gets its match id appended so the two matches' intermediate
# artifacts (and their full chapter export directories) never collide.
SUFFIX = '' if MATCH_ID == '10508' else f'_{MATCH_ID}'


def pkl_path(name):
    """name without extension, e.g. pkl_path('match_features') ->
    '<output>/match_features.pkl' (10508) or '<output>/match_features_10515.pkl'."""
    os.makedirs(OUT_ROOT, exist_ok=True)
    return os.path.join(OUT_ROOT, f'{name}{SUFFIX}.pkl')


def json_path(name):
    os.makedirs(OUT_ROOT, exist_ok=True)
    return os.path.join(OUT_ROOT, f'{name}{SUFFIX}.json')


CHAPTERS_DIR = os.path.join(OUT_ROOT, f'chapters{SUFFIX}')

DEADBALL_SETPIECE_TYPES = {'C', 'F', 'P'}  # corner, free kick, penalty
MASK_BEFORE_S = 2.0
MASK_AFTER_S = 15.0


def load_roster():
    with open(ROSTER_PATH) as f:
        roster = json.load(f)
    jersey_lookup = {}       # (team_id_str, shirt_str) -> (nickname, posgroup)
    player_id_lookup = {}    # player_id_str -> (nickname, posgroup, team_id_int)
    for r in roster:
        jersey_lookup[(r['team']['id'], r['shirtNumber'])] = (r['player']['nickname'], r['positionGroupType'])
        player_id_lookup[str(r['player']['id'])] = (r['player']['nickname'], r['positionGroupType'], int(r['team']['id']))
    return roster, jersey_lookup, player_id_lookup


def load_metadata():
    with open(METADATA_PATH) as f:
        meta = json.load(f)[0]
    return meta


def orientation_lookup(meta):
    """period -> True if HOME team defends the positive-x goal in that period."""
    home_start_left = meta['homeTeamStartLeft']
    home_start_left_et = meta.get('homeTeamStartLeftExtraTime')
    if home_start_left_et is None:          # key absent, or present but null (matches without extra time)
        home_start_left_et = home_start_left
    return {
        1: (not home_start_left),
        2: home_start_left,
        3: (not home_start_left_et),
        4: home_start_left_et,
    }


def pitch_dims(meta):
    try:
        length = meta['stadium']['pitches'][0]['length']
        width = meta['stadium']['pitches'][0]['width']
    except (KeyError, IndexError, TypeError):
        length, width = 105.0, 68.0
    return length, width


if __name__ == '__main__':
    roster, jersey_lookup, player_id_lookup = load_roster()
    meta = load_metadata()
    print(f"Players in roster: {len(roster)}")
    print(f"Home defends positive x by period: {orientation_lookup(meta)}")
    print(f"Pitch dims: {pitch_dims(meta)}")
    print(f"FPS: {meta.get('fps')}")
