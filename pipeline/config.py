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

MATCHES = {
    '10508': dict(
        game_dir='/home/claude/wc2022',
        home_team_id=374,   # Morocco
        away_team_id=52,    # Spain
        home_team_name='Morocco',
        away_team_name='Spain',
        home_color='#e8323a', home_color_dim='#7a2226',   # Morocco red (unchanged)
        away_color='#4fd8d0', away_color_dim='#235e59',   # Spain teal (unchanged)
        head_template='/home/claude/project_work/artifact_head.html',
        label='Morocco vs Spain (Round of 16)',
        heatmap_vmax_log=5.1933,  # round 37 recalibration -- see full_match_heatmap.py docstring
    ),
    '10515': dict(
        game_dir='/home/claude/wc2022_10515',
        home_team_id=363,   # France (home team per metadata)
        away_team_id=374,   # Morocco
        home_team_name='France',
        away_team_name='Morocco',
        home_color='#2f6fe0', home_color_dim='#1f3a66',   # France blue
        away_color='#e8323a', away_color_dim='#7a2226',   # Morocco red (same identity as 10508)
        head_template='/home/claude/project_work/artifact_head_10515.html',
        label='Morocco vs France (Semifinal)',
        heatmap_vmax_log=5.3972,  # round 37 recalibration
    ),
    '3821': dict(
        game_dir='/home/claude/wc2022_3821',
        home_team_id=368,   # Germany (home team per metadata)
        away_team_id=57,    # Japan
        home_team_name='Germany',
        away_team_name='Japan',
        home_color='#f2c14e', home_color_dim='#7d6320',   # Germany gold
        away_color='#3399ff', away_color_dim='#1f4d80',   # Japan blue
        head_template='/home/claude/project_work/artifact_head_3821.html',
        label='Japan vs Germany (Group Stage)',
        heatmap_vmax_log=5.6386,  # round 37 recalibration
    ),
    '3823': dict(
        game_dir='/home/claude/wc2022_3823',
        home_team_id=362,   # Belgium (home team per metadata)
        away_team_id=380,   # Canada
        home_team_name='Belgium',
        away_team_name='Canada',
        home_color='#e33b4d', home_color_dim='#6b1620',   # Belgium red
        away_color='#7c9fff', away_color_dim='#2a3866',   # Canada periwinkle (actual kit is white/red -- white unreadable on the dark pitch, red would clash with Belgium's, so substituted like Germany's gold was)
        head_template='/home/claude/project_work/artifact_head_3823.html',
        label='Belgium vs Canada (Group Stage)',
        heatmap_vmax_log=5.1722,  # round 37 recalibration
    ),
    '10517': dict(
        game_dir='/home/claude/wc2022_10517',
        home_team_id=364,   # Argentina (home team per metadata)
        away_team_id=363,   # France
        home_team_name='Argentina',
        away_team_name='France',
        home_color='#5fb8ea', home_color_dim='#1d3f52',   # Argentina sky blue (actual kit is white/sky-blue stripes -- white unreadable on the dark pitch)
        away_color='#e63946', away_color_dim='#712024',   # France red (actual kit is navy -- too close to Argentina's blue side by side, substituted with French-flag red for contrast)
        head_template='/home/claude/project_work/artifact_head_10517.html',
        label='Argentina vs France (Final)',
        heatmap_vmax_log=6.2112,  # round 37 recalibration -- highest of the five, this match runs the "hottest"
    ),
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
HEATMAP_VMAX_LOG = _match.get('heatmap_vmax_log', 4.375829836329568)

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
    '.../match_features.pkl' (10508) or '.../match_features_10515.pkl'."""
    return f'/home/claude/project_work/{name}{SUFFIX}.pkl'


def json_path(name):
    return f'/home/claude/project_work/{name}{SUFFIX}.json'


CHAPTERS_DIR = f'/home/claude/project_work/chapters{SUFFIX}'

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
    home_start_left_et = meta.get('homeTeamStartLeftExtraTime', home_start_left)
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
