# Checking the rebuilt analysis code

The analysis views, the comparison board, the scouting-report chips and the hub builder were rewritten after
the original working copy was lost. The published outputs survived, so every rebuilt script was checked
against them: same definitions, same layout, same numbers where possible. This page records how close each
one gets.

Matches checked: Germany vs Japan (3821), Belgium vs Canada (3823), Argentina vs France (10517) and Köln vs
Bayern Munich (182265). Morocco vs Spain (10508) and France vs Morocco (10515) will be checked once their raw
files are back.

## Numbers

| What | Rebuilt vs published |
|---|---|
| Live minutes per match | 3823 59.3 vs 59.2 · 3821 59.2 vs 59.0 · 10517 71.3 vs 71.3 · 182265 52.7 vs 53.3 |
| Possession share (comparison board) | exact for all 8 team-matches |
| Where they attacked | final-third entries exact; lane shares within 2 points |
| In and out of possession | every summary sentence exact ("back line drops 3 m, block narrows 6 m ...") |
| Shape over time | first and last 15 minutes within 1 m; match medians within 0.3 m |
| Where the gaps open | holes within 2 m², free opponents within 0.01, most frequent hole exact |
| Line-breaking passes | counts within 2 (Köln conceded 49 vs 51); the same leading passers (ties aside), their counts within 2 |
| Turnovers | 3821 and 3823 exact except Belgium's fast attacks (22 vs 21); the final within 1; Köln within 3 |
| Chances conceded | every count exact except one: Japan's shots after a ball win in its own third (5 vs 6) |
| Pressing | presses within 4, success rate within 1 point; Köln waits on the pooled threshold |
| Comparison board | as the views above; possession exact, back line, length and width within 0.3 m |

## Page builders

| What | Result |
|---|---|
| Scouting-report chips | from the published board and press rates, the rank logic reproduces all 60 published chips exactly |
| Hub page | from the published data files, `build_match_hub.py` reproduces the live hub exactly (page and data) |
| Chapter head template | `make_head.py 182265` is byte-identical to the head of the published Köln chapters |
| Chapter tail template | recovered from a published chapter (it carries the team-colour fix the backup lacked) |
| Images | compared side by side with the published ones: same layout, titles, legends and footnotes |

## Why the small differences

- The definitions are the published ones, but some edge decisions had to be re-made: where a possession
  flicker shorter than 1 s is absorbed, which frame a 5 Hz sample lands on, and how ties at a threshold are
  broken. Each moves a count by one here and there.
- Köln vs Bayern differs a little more everywhere (0.6 live minutes fewer, a few more ball wins). Its data
  was converted again from freshly downloaded DFL files, so the inputs are probably not byte-identical to
  the originals.
- The chances view counts the third a ball was won in from the shooting team's side, as the published
  captions do; `turnover_thirds_lost` gives the same turnovers from the defending team's side.
