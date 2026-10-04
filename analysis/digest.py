"""Index of the analysis images for the Match Library hub: one entry per view per match, with alt text.

    python3 analysis/digest.py [<match id> ...]       # default: every match in the registry

Writes output/analysis/an_kinds.json: the views in the hub's tab order as {name, desc, data}, where data holds an
entry {slug, label, cap, src, w, h, alt} for each match whose image exists. The alt text describes the image and
quotes its key numbers, taken from the view's JSON, so a screen-reader user gets the same headline facts.
"""
import json
import os
import sys

from PIL import Image

from common import R, out_path

# hub tab name, image folder, the one-line description shown above the image
KINDS = [
    ('Where they attacked', 'attack', 'Where each team had the ball, and how much of its forward movement went down the left wing, through the middle or down the right wing.'),
    ('Team heatmap', 'teams', 'Where each team’s outfield players spent the game as a whole, with every starter’s average position marked by shirt number.'),
    ('In & out of possession', 'shape', 'Each team’s shape with the ball next to its shape without it. Lines on the right-hand pitch show how far each starter moved when the ball was lost; underneath: team centre, back-line height, width and length.'),
    ('Shape over time', 'time', 'How each team’s block without the ball changed as the game went on: back-line height, length and width, live play only, with the goals marked. The header compares the first 15 minutes with the last 15 of normal time.'),
    ('Where the gaps open', 'gaps', 'Each team’s block seen from behind its own back line, with every defending moment lined up on the back line. Amber shows the holes that kept opening inside the block, and the right-hand view shows where opponents stood free inside it. Circles mark starters’ usual spots.'),
    ('Line-breaking passes', 'lb', 'Completed passes that went through a team’s midfield or back line, bypassing three or more of its deepest seven outfielders. Arrows in the attacking team’s colour were received in front of the back line; white ones got in behind it. On the right: who played them, and which channel they arrived in.'),
    ('Chances conceded', 'chances', 'Every shot each team faced, rewound to the start of the attack (up to 20 seconds). Each small pitch shows the defenders at the shot with a line back to where they stood when the attack began, the attackers, and the ball’s path to the shot. Goals are circled.'),
    ('Turnovers', 'turn', 'Where each team won and lost the ball in open play, and what followed: bright dots are turnovers where the team that won it was in or into the final third within 10 seconds, diamonds led to a shot within 15. Restarts after the ball went out are left out.'),
    ('Pressing', 'press', 'Where each team got tight on the ball carrier, what set it off (a counter-press straight after losing the ball, a back pass, a pass out wide), and whether it won the ball back within 5 seconds. On the right: success rate by trigger and by player.'),
    ('Player heatmaps', 'players', 'Where each player spent the game, grouped by team and position: starters first, then subs who played 5+ minutes.'),
]


def num(v):
    """Short number for prose: whole numbers without a decimal point, others to 2 places at most."""
    return str(int(v)) if float(v).is_integer() else f'{v:.2f}'.rstrip('0')


def alt_attack(t, d, h, a):
    def one(n):
        s = d[n]
        return (f"{n}: left {round(s['left'])}%, middle {round(s['middle'])}%, right {round(s['right'])}%, "
                f"{s['entries']} final-third entries")
    return f'Where they attacked, {t}. {one(h)}; {one(a)}.'


def alt_teams(t, d, h, a):
    return (f"Team heatmaps, {t}: where each team's outfield players spent the match, "
            "with every starter's average position.")


def alt_players(t, d, h, a):
    return (f'Player heatmaps, {t}: one small pitch per player showing where he spent the match, '
            'with his median position.')


def alt_shape(t, d, h, a):
    return f"In and out of possession, {t}. {h}: {d[h]['line']}. {a}: {d[a]['line']}."


def alt_time(t, d, h, a):
    def one(n):
        return (f"{n}: back line {round(d[n]['first15']['line'])} m in the first 15 minutes, "
                f"{round(d[n]['last15']['line'])} m in the last 15 of normal time")
    return (f"Shape over time, {t}: line charts of each team's back-line height, block length and width without "
            f'the ball through the match. {one(h)}. {one(a)}.')


def alt_gaps(t, d, h, a):
    def one(n):
        s = d[n]
        return (f"{n} defending: {s['holes']} square metres of holes on average, {num(s['free'])} opponents free "
                f"inside the block, most frequent hole {s['biggest_pocket']}")
    return (f"Where the gaps open, {t}: each team's defensive block seen from behind its back line, with its "
            f'recurring holes and the opponents free inside it. {one(h)}. {one(a)}.')


def alt_lb(t, d, h, a):
    def one(n):
        s = d[n]
        return (f"Against {n}: {s['conceded']}, {s['behind']} received behind the back line; "
                f"most by {', '.join(s['top_passers'][:3])}")
    return (f"Line-breaking passes, {t}: arrows of every completed pass through each team's midfield or back line. "
            f'{one(h)}. {one(a)}.')


def alt_chances(t, d, h, a):
    def one(n):
        s = d[n]
        return (f"{n} faced {s['shots_faced']} shots: {s['after_turnover']} after losing the ball, "
                f"{s['after_restart']} after restarts, {s['long_possession']} from sustained possession")
    return (f'Chances conceded, {t}: one small pitch per shot faced, traced back to the start of the attack. '
            f'{one(h)}. {one(a)}.')


def alt_turn(t, d, h, a):
    def one(n):
        s = d[n]
        return (f"{n} won it {s['won']} times, {s['won_opp_half']} in the opponents' half; "
                f"{s['won_to_final_third_10s']} were in or into the final third inside 10 s and "
                f"{s['won_to_shot_15s']} led to a shot")
    return f'Turnovers, {t}: pitch maps of where each team won and lost the ball. {one(h)}. {one(a)}.'


def alt_press(t, d, h, a):
    def one(n):
        s = d[n]
        return (f"{n}: {s['presses']} presses, {s['win_rate']}% won the ball within 5 s, "
                f"{s['in_opp_half']} in the opponents' half")
    return (f'Pressing, {t}: where each team pressed and whether it won the ball within 5 seconds. '
            f'{one(h)}. {one(a)}.')


ALT = {'attack': alt_attack, 'teams': alt_teams, 'players': alt_players, 'shape': alt_shape, 'time': alt_time,
       'gaps': alt_gaps, 'lb': alt_lb, 'chances': alt_chances, 'turn': alt_turn, 'press': alt_press}


def entry(mid, kind):
    m = R.get(mid)
    img, data = (out_path('analysis', kind, f"{m['slug']}.{ext}") for ext in ('jpg', 'json'))
    if not (os.path.exists(img) and os.path.exists(data)):
        return None
    w, h = Image.open(img).size
    d = json.load(open(data, encoding='utf-8'))
    alt = ALT[kind](m['title'], d, m['home']['name'], m['away']['name'])
    return dict(slug=m['slug'], label=m['title'], cap=m['sub'], src=f"{kind}/{m['slug']}.jpg", w=w, h=h, alt=alt)


def main(mids):
    out, missing = [], []
    for name, kind, desc in KINDS:
        data = []
        for mid in mids:
            e = entry(mid, kind)
            if e:
                data.append(e)
            else:
                missing.append(f'{kind}/{R.get(mid)["slug"]}')
        out.append(dict(name=name, desc=desc, data=data))
    path = out_path('analysis', 'an_kinds.json')
    json.dump(out, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f"{sum(len(k['data']) for k in out)} images indexed -> {path}")
    if missing:
        print('not built yet:', ', '.join(missing))


if __name__ == '__main__':
    main(sys.argv[1:] or R.ids())
