# Methodology

How the model turns raw tracking data into a defensive read, what each number means, and how each piece was checked. For the research behind it, see [LITERATURE.md](LITERATURE.md).

## Data

- **PFF FC 2022 FIFA World Cup tracking.** Free, request-based dataset covering all 64 matches. Player and ball positions are derived from the TV broadcast at about 30 frames a second, with an embedded event feed.
  - Five matches are processed: Morocco vs Spain (R16), France vs Morocco (SF), Germany vs Japan (group), Belgium vs Canada (group) and Argentina vs France (final).
  - Because the tracking comes from broadcast video, players off-screen are estimated and the ball has gaps. Ball coverage ranges from 10% to 90% across five-minute stretches. Every chapter shows its coverage, and frames without a ball are flagged on screen.
- **DFL Bundesliga open data** (Bassek et al., 2025, CC-BY 4.0). Optical in-stadium tracking at 25 Hz, plus the DFL event feed. It enters the pipeline through a converter (`providers/dfl_to_pff.py`); see below. One match is processed: Köln vs Bayern Munich, 27 May 2023.

## Cleaning the tracking (`animation_export.py`)

Raw broadcast tracking has two kinds of defect:
- single-frame detector spikes, rejected and bridged
- constant small-scale jitter

A constant-velocity **Kalman filter with an RTS smoother** replaced an earlier ad hoc smoothing stack. It cut frame-to-frame acceleration noise about 5x compared with that stack and about 30x compared with the raw data. Gaps longer than the filter can safely bridge stay as gaps.

## Team shape (`tracking_features.py`)

Computed per frame from outfield positions, independent of possession:

- **Depth:** mean distance from own goal of the back line, the 4 deepest outfield players. An earlier version averaged all 10 outfielders, which let forwards drag the "back line" 12-20 m deeper than any actual defender.
- **Inter-line (team length):** distance between the deepest and the most advanced outfield player.
- **Back-line gap:** the largest lateral gap between neighbours in the back four.

The classifier uses a 3-second rolling mean so labels don't flicker. The on-screen band and numbers are recomputed live from the drawn player positions, so the band always sits exactly on the deepest defender and the most advanced player.

## Defensive phase (`classification.py`, `transitions.py`)

Fixed, literature-anchored thresholds, identical in every match. They deliberately aren't clustered per match: per-match clustering was unstable, and fixed cut-offs mean the same thing everywhere.

| Phase | Rule |
|---|---|
| High Press | back-line depth ≥ 40 m from own goal |
| Mid Block | 32 m ≤ depth < 40 m |
| Low Block | depth < 32 m **and** team length ≤ 20 m (otherwise Mid Block) |
| Transition | detected, not thresholded: after a possession change, until depth and team length settle to a new value |

## Pressure on the ball carrier (`pressure.py`)

A single-target simplification of the time-to-intercept idea in Bekkers (2025):

- `time_to_intercept = reaction (0.7 s) + |ball − (defender + velocity × 0.7 s)| / max_speed`
- **Per-defender contribution:** `1 / (1 + exp((t − 1.5) / 0.5))`
- **Frame score:** the probabilistic OR across defenders, `1 − Π(1 − contribution)`
- **Max speed:** each player's own 99th-percentile speed. Where the data has no speed field (three of the World Cup matches), it's estimated from frame-to-frame displacement of the smoothed positions.

**Check:** on moments PFF analysts tagged as "under pressure", the mean score is 0.739. On untagged moments it's 0.538.

### Engaged vs passive (`engagement.py`)

The same score, smoothed over 3 s, is split at a per-match threshold chosen with Youden's J against PFF's analyst pressure tags:

| Match | Threshold | AUC |
|---|---|---|
| Morocco vs Spain | 0.64 | 0.766 |
| France vs Morocco | 0.67 | 0.778 |
| Germany vs Japan | 0.67 | 0.709 |
| Belgium vs Canada | 0.72 | 0.726 |
| Argentina vs France | 0.71 | 0.677 |

DFL data has no pressure tags, so a DFL match uses the median of these thresholds (0.67).

## Tight vs loose marking (`analysis_ball_proximal_compactness.py`, `compactness_tag.py`)

Ball-proximal compactness is the largest distance among the 5 outfield defenders nearest the ball. It was tested as a possible third defining variable against an independent outcome: does the ball get won back within 5, 8 or 10 seconds? It wasn't adopted, for two reasons:
- There was no clean threshold.
- Once team length is known, it adds no predictive value (checked with partial correlation).

It ships as a descriptive TIGHT/LOOSE tag, split at the 23 m median.

## Danger heatmap (`heatmap_export.py`, `full_match_heatmap.py`)

**Dangerous Accessible Space** (Bischofberger & Baca, 2026) combines two things:
- A simulated pass-completion map from the current positions and speeds.
- An expected-goals-style value for each location, based on distance and angle to goal.

How it's rendered:
- The grid is 52 × 34 cells at 5 Hz.
- Each frame is Gaussian-blurred, which fixes gaps between the simulation's rays.
- It's log-scaled with `log1p(mass / floor)`, because raw values span about 8 orders of magnitude.
- Brightness is capped at each match's own 99.5th percentile. A single ceiling taken from one 5-minute window had over-saturated every full match.

## Match-level analysis (`analysis/`)

Ten views read a whole match at once. Each team is drawn attacking left to right, and each script's docstring
gives its full definitions.

- **Live play** (`inplay.py`): from an on-the-ball event until the next, if the next starts within 5 s of the
  last touch and the ball didn't go out in between; otherwise until 1.5 s after the touch. 53-77 live minutes
  per match.
- **Where they attacked:** the share of the ball's forward movement down each third of the pitch's width, in
  the middle and final thirds while in possession, plus final-third entries.
- **Team and player heatmaps; in and out of possession:** 5 Hz positions, keepers left out of team views;
  back line = mean distance of the deepest four outfielders from their own goal; width and length = the
  outfield spread per moment, as medians.
- **Shape over time:** back-line height, block length and width without the ball, live play only, as a
  10-minute rolling median.
- **Where the gaps open:** every live defending moment lined up on the back line. A hole is a 1 m cell inside
  the outfielders' convex hull with no defender within 8 m; a free opponent stands inside the block with no
  defender within 5 m.
- **Line-breaking passes:** a completed pass or cross in live play, moving the ball 5 m or more towards goal,
  that bypasses three or more of the defending team's deepest seven outfielders.
- **Turnovers:** a switch of possession (flickers under 1 s absorbed) with the ball live for the 2 s before
  it. Fast = in or into the final third within 10 s; diamonds led to a shot within 15 s.
- **Chances conceded:** every shot faced, rewound to the start of the attack (the shooting team's possession
  start or the last restart, at most 20 s back) and classed as after a turnover, after a restart or long
  possession.
- **Pressing:** pressure on the carrier above the match's engaged threshold, by the defending team, for at
  least 1/3 s (gaps under 1 s merged); won = the ball changes hands within 5 s; triggers are counter-press,
  back pass, pass out wide, other pass, or carry.
- **Comparison board and scouting reports:** the board takes one row per team-match from the views; the
  reports are written by hand from the views' numbers, with five stat chips ranked across every team.

## Cross-checks

- **PPDA:** passes allowed per defensive action, from events only. Morocco 18.44 (passive), Spain 4.04 (aggressive). Restricted to organised-defence time, PPDA rises from High Press to Mid Block to Low Block for both teams. That independent event-based measure agrees with the tracking-based phases.
  - Getting there meant fixing a clock bug: frame-count "video time" drifts up to 12.5 minutes from the real match clock by extra time.
- **Wide pressure → back-line height** (a coaching hypothesis). In organised defence, sustained pressure on the ball out wide comes before a slightly deeper back line 13-15 s later in both matches tested (|r| = 0.11 and 0.08). Real but modest. Whole-match and per-team versions don't replicate.

## Running the model on broadcast footage (`footage/`)

A prototype on the first 6 minutes of France vs Morocco:
- tiled YOLO detection
- camera calibration from pitch markings, with a fixed camera position solved from one frame and per-frame pan, tilt and zoom
- kit-colour team split and tracking
- the same definitions as above

It reads a phase only when all 10 defending outfielders are in shot. Agreement with PFF:

| Measure | Value |
|---|---|
| Median player position error | 0.83 m |
| PFF players in shot found | 84% |
| Pressure correlation | 0.74 |
| Engaged/passive agreement | 85% |
| Possession agreement (after a ball-tracking rework) | 91% |

## Other data providers (`providers/`)

`dfl_to_pff.py` converts DFL open data into PFF's file layout, so the rest of the pipeline runs unchanged:
- **Positions:** resampled from 25 to 29.97 fps. Several pipeline thresholds assume PFF's frame rate.
- **Speed:** km/h converted to m/s.
- **Event timing:** each event is snapped to the moment the tracked ball was closest to the acting player, after a global clock offset (−0.44 s for Köln vs Bayern).
- **On-the-ball spells:** start where the ball first stays within 2 m of the player.
- **Ball out of play:** taken from DFL's own ball-status flag.

**Check (Köln vs Bayern):**
- The possession timeline agrees with DFL's own per-frame possession flag on 87% of live frames.
- All 21 shots and 3 goals come through.
- Attack directions from metadata, shots and keeper positions agree.

## Known limits

- The pressure score is evaluated at the ball only, not over a full pitch-control surface.
- The phase thresholds are literature-anchored, not fitted to outcomes.
- The TIGHT/LOOSE split is descriptive, not validated against outcomes.
- In the World Cup data, about half of off-screen player positions are PFF estimates, and the animation draws them like observed ones.
- Six matches is still a small sample for comparing teams.
