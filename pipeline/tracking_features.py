"""
Full-match pass over the tracking file producing per-frame scalar features:
depth, inter-line compactness, and back-line gap size for both teams, plus
ball position. Uses Kalman-smoothed positions (less jitter than raw).

Player velocity (for the pressure metric) is derived separately, on-demand,
for the short animation window only -- no need to carry per-player vectors
across the whole match.
"""
import bz2
import json
import pandas as pd
from config import (TRACKING_PATH, HOME_TEAM_ID, AWAY_TEAM_ID, load_roster)


def build_jersey_lookups(jersey_lookup):
    home_lookup = {j: v for (t, j), v in jersey_lookup.items() if t == str(HOME_TEAM_ID)}
    away_lookup = {j: v for (t, j), v in jersey_lookup.items() if t == str(AWAY_TEAM_ID)}
    return home_lookup, away_lookup


def _team_depth_and_lines(players, lookup, own_goal_x, n_backline=4):
    """players: list of {jerseyNum,x,y,...} (smoothed). Returns (depth, inter_line,
    backline_gap, outfield_count) excluding GK.

    depth = mean distance from own goal of the back line ONLY (the n_backline
    deepest outfield players at that instant, not the whole team) -- this is
    the number the canvas draws as the dashed "back line" and the number the
    High Press/Mid/Low Block thresholds are checked against, so it needs to
    actually track the last line of defenders. It used to average ALL 10
    outfield players including forwards, which the user's own framing (this
    should show "the highest and lowest defender") and a direct check both
    called out as wrong: during a Defensive Transition frame with home_depth
    reported as ~55m, the real back four (H6/H5/H2/H4) were sitting at
    34-43m -- the reported "back line" was 12-20m deeper than every actual
    defender, dragged there by forwards who hadn't dropped back yet. Fixed by
    computing depth from the same back_group already used for backline_gap
    below, instead of the full outfield list.

    inter_line = team length: the deepest player's distance from own goal
    minus the shallowest player's (i.e. furthest defender to striker), still
    computed across the WHOLE outfield team on purpose -- confirmed with the
    user this is the intended definition (not a back-line-only spread), and
    it's a standard compactness/block-height measure in the tactical
    literature. Left unchanged.

    backline_gap = largest lateral (y) gap among the deepest n_backline
    outfield players (a generalized 'back line' proxy), unchanged.
    """
    depths = []
    ys_by_depth = []
    seen_jerseys = set()
    for pl in players:
        # some frames carry duplicate entries per jersey (observed data quirk --
        # PFF notes frames with multiple possession events can be duplicated);
        # keep only the first occurrence per jersey per frame.
        jn = pl['jerseyNum']
        if jn in seen_jerseys:
            continue
        info = lookup.get(jn)
        if info is None or pl.get('x') is None:
            continue
        seen_jerseys.add(jn)
        _, posgroup = info
        if posgroup == 'GK':
            continue
        d = abs(own_goal_x - pl['x'])
        depths.append(d)
        ys_by_depth.append((d, pl['y']))

    if len(depths) < 3:
        return None, None, None, len(depths)

    inter_line = max(depths) - min(depths)

    # back-line group: the n_backline deepest players (closest to own goal).
    # Used for both the back-line depth (below) and the lateral gap.
    ys_by_depth.sort(key=lambda t: t[0])  # smallest depth-from-goal = deepest defender
    back_group = ys_by_depth[:min(n_backline, len(ys_by_depth))]
    depth = sum(d for d, _ in back_group) / len(back_group)
    back_ys = sorted(y for _, y in back_group)
    backline_gap = max((back_ys[i + 1] - back_ys[i] for i in range(len(back_ys) - 1)), default=0.0)

    return depth, inter_line, backline_gap, len(depths)


def extract_match_features(orientation, jersey_lookup, mask_windows_fn=None, fps=29.97):
    home_lookup, away_lookup = build_jersey_lookups(jersey_lookup)
    rows = []

    with bz2.open(TRACKING_PATH, 'rt') as f:
        for line in f:
            d = json.loads(line)
            period = d['period']
            home_positive = orientation.get(period)
            if home_positive is None:
                continue

            frame = d['frameNum']
            video_t = frame / fps

            home_own_goal_x = 52.5 if home_positive else -52.5
            away_own_goal_x = -52.5 if home_positive else 52.5

            home_players = d.get('homePlayersSmoothed') or d.get('homePlayers') or []
            away_players = d.get('awayPlayersSmoothed') or d.get('awayPlayers') or []

            home_depth, home_il, home_gap, home_n = _team_depth_and_lines(home_players, home_lookup, home_own_goal_x)
            away_depth, away_il, away_gap, away_n = _team_depth_and_lines(away_players, away_lookup, away_own_goal_x)
            if home_depth is None or away_depth is None:
                continue

            ball = d.get('ballsSmoothed')
            if isinstance(ball, list):
                ball = ball[0] if ball else None
            # ballsSmoothed can be a non-None dict with x=None (not tracked this
            # frame) -- must check the actual coordinate, not just truthiness,
            # before falling back to the raw ball.
            if ball is None or ball.get('x') is None:
                raw_ball = d.get('balls') or []
                raw_ball0 = raw_ball[0] if raw_ball else None
                if raw_ball0 is not None and raw_ball0.get('x') is not None:
                    ball = raw_ball0
            bx, by, bz = (ball.get('x'), ball.get('y'), ball.get('z')) if ball else (None, None, None)

            rows.append((frame, period, video_t, home_depth, away_depth, home_il, away_il,
                         home_gap, away_gap, home_n, away_n, bx, by, bz))

    df = pd.DataFrame(rows, columns=['frameNum', 'period', 'video_t', 'home_depth', 'away_depth',
                                      'home_inter_line', 'away_inter_line', 'home_backline_gap',
                                      'away_backline_gap', 'home_n_tracked', 'away_n_tracked',
                                      'ball_x', 'ball_y', 'ball_z'])
    df = df.sort_values('video_t').reset_index(drop=True)
    return df


if __name__ == '__main__':
    from config import load_metadata, orientation_lookup, pkl_path
    roster, jersey_lookup, player_id_lookup = load_roster()
    meta = load_metadata()
    orientation = orientation_lookup(meta)

    import time
    t0 = time.time()
    df = extract_match_features(orientation, jersey_lookup, fps=meta['fps'])
    print(f"Extracted {len(df)} frames in {time.time()-t0:.1f}s")
    print(df.describe().round(2).to_string())
    out = pkl_path('match_features')
    df.to_pickle(out)
    print(f"Saved {out}")
