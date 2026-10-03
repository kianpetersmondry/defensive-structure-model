# Data providers

The pipeline reads PFF FC's file layout: `metadata.json`, `roster.json`, `events.json` and `tracking.jsonl.bz2`. A converter here turns another provider's data into those four files, and everything downstream runs unchanged.

## DFL (Bundesliga open data)

Source: Bassek et al. (2025), *Scientific Data*; files on [figshare](https://doi.org/10.6084/m9.figshare.28196177); CC-BY 4.0, © DFL. Each match needs three XML files: match information, raw events and raw observed positions (~400 MB).

```
python3 providers/dfl_to_pff.py <dfl folder> DFL-MAT-XXXXXX <match id> <out folder> \
    --home-name "Köln" --away-name "Bayern Munich" --home-code KOE --away-code FCB
```

What it does:
- **Positions:** 25 Hz, metres, centred, the same axes as PFF. They're resampled to 29.97 fps, because several pipeline thresholds assume PFF's frame rate.
- **Speed:** DFL gives km/h; it becomes PFF `speed` in m/s.
- **Event timing:** DFL events are single, hand-typed timestamps. After one global clock offset, each is snapped to the frame within ±0.6 s where the ball was closest to the acting player.
- **On-the-ball spells:** a player's spell starts where the ball first stays within 2 m of him, and runs until he passes or shoots.
- **Ball out of play:** DFL's per-frame ball status becomes OUT events.
- **Tackles:** filed as challenges under the ball carrier's spell.

DFL has no analyst "under pressure" tag and no clearance event. The engaged/passive threshold then falls back to the median of the PFF-calibrated matches.

Checked on Köln vs Bayern (DFL-MAT-J03WMX):
- The possession timeline agrees with DFL's own per-frame possession flag on 87% of live frames.
- All 21 shots and 3 goals come through.
- Attack directions from metadata, shots and keepers agree.
