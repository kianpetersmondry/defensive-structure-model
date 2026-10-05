# Checking the rebuilt analysis code

The analysis views, the comparison board, the scouting-report chips and the hub builder were rewritten after
the original working copy was lost. The published outputs survived, so every rebuilt script was checked
against them: same definitions, same layout, same numbers where possible. This page records how close each
one gets.

All six matches were checked: Morocco vs Spain (10508), France vs Morocco (10515), Germany vs Japan (3821),
Belgium vs Canada (3823), Argentina vs France (10517) and Köln vs Bayern Munich (182265).

## Numbers

| What | Rebuilt vs published |
|---|---|
| Live minutes per match | 10508 77.0 vs 77.4 · 10515 60.2 vs 60.4 · 3821 59.2 vs 59.0 · 3823 59.3 vs 59.2 · 10517 71.3 vs 71.3 · 182265 52.7 vs 53.3 |
| Engaged/passive thresholds | exact: 10508 0.64 (AUC 0.766), 10515 0.67 (0.778), 3821 0.67, 3823 0.72, 10517 0.71; Köln's pooled 0.67 |
| Possession share (comparison board) | exact for all 12 team-matches |
| Where they attacked | final-third entries exact; lane shares within 2 points |
| In and out of possession | every summary sentence exact ("back line drops 3 m, block narrows 6 m ...") |
| Shape over time | first and last 15 minutes within 1 m; match medians within 0.4 m |
| Where the gaps open | holes within 2 m², free opponents within 0.04, most frequent hole exact |
| Line-breaking passes | counts within 2 (Köln conceded 49 vs 51); the same leading passers (ties aside), their counts within 2 |
| Turnovers | 3821 and 3823 exact except Belgium's fast attacks (22 vs 21); 10508, 10515 and the final within 1; Köln within 3 |
| Chances conceded | every count exact except one: Japan's shots after a ball win in its own third (5 vs 6) |
| Pressing | presses within 6 (Morocco vs Spain 412 vs 406), success rate within 2 points |
| Comparison board | as the views above; possession exact, back line, length and width within 0.4 m |

## Page builders

| What | Result |
|---|---|
| Scouting-report chips | from the published board and press rates, the rank logic reproduces all 60 published chips exactly; from the rebuilt numbers, 53 of the 60 ranks are unchanged and none moves more than two places |
| Hub page | from the published data files, `build_match_hub.py` reproduces the live hub exactly (page and data) |
| Chapter head template | `make_head.py 182265` is byte-identical to the head of the published Köln chapters |
| Chapter tail template | recovered from a published chapter (it carries the team-colour fix the backup lacked) |
| Images | compared side by side with the published ones: same layout, titles, legends and footnotes |

## Why the small differences

- The definitions are the published ones, but some edge decisions had to be re-made: where a possession
  flicker shorter than 1 s is absorbed, which frame a 5 Hz sample lands on, and how ties at a threshold are
  broken. Each moves a count by one here and there.
- The hand-written report text quotes the published numbers, so a few quoted figures sit a point or two away
  from a rebuilt run (Spain's press success is 41% rebuilt against the 39% the report quotes).
- Köln vs Bayern differs a little more everywhere (0.6 live minutes fewer, a few more ball wins). Its data
  was converted again from freshly downloaded DFL files, so the inputs are probably not byte-identical to
  the originals.
- The chances view counts the third a ball was won in from the shooting team's side, as the published
  captions do; `turnover_thirds_lost` gives the same turnovers from the defending team's side.

## The six further DFL matches

Bochum vs Bayer Leverkusen (182269) and Fortuna Düsseldorf's home games against Jahn Regensburg (182321),
Hansa Rostock (182338), Nürnberg (182374), St. Pauli (182402) and Kaiserslautern (182421) were built with the
rebuilt code from the start, so there is no published version to compare with. They were checked against
the DFL record instead:

| Check | Result |
|---|---|
| Final score | all six match DFL's result (3-0, 4-0, 3-1, 0-1, 1-0, 1-2) |
| Goals | every goal found in the tracking at the right minute, 16 of 16 |
| Shots per team | the chances view matches the DFL event feed for all twelve teams |
| Attack directions | from the shots, agreeing with the start sides and the keepers; no keeper swaps |
| Chapters | 18 or 19 per match, ball tracked in 100% of frames |
| Engaged/passive threshold | pooled, 0.67 (DFL has no pressure tags) |
| Heatmap ceiling | 5.72 to 6.29, computed by the new `ceiling` step; the same step gives 5.64 for Köln vs Bayern against its published 5.65 |

Bochum's keeper standing 26 m off his line at kick-off is real: Bochum lined up across halfway and kicked
long into Leverkusen's corner, and his match median sits 37 m from goal as normal.

Adding them took the board from 12 to 24 teams. The ranking words in the twelve published reports were
re-checked against all 24 and 26 of them changed (Köln's block is no longer the longest, Bayern no longer
has the most open space, and so on); the published numbers themselves are unchanged.

