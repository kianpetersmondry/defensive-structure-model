# Footage overlay prototype — code

Runs the defensive-structure model on real broadcast video and draws the
read onto the players. Built and tested on the first ~6 minutes of France vs
Morocco (2022 World Cup semifinal). Results: see `../docs/METHODOLOGY.md`,
"Running the model on broadcast footage".

Rule: a phase, the inter-line band and depth / team length are shown only when exactly 10 defending outfield players are in shot (`FULL_TEAM` in `model_footage.py`).

Order to run (from this folder, with the video saved as `clip.mp4`):

1. `run_detect.py`    — player/ball detection, every 3rd frame -> detections.pkl
2. `run_calib.py`     — camera pan/tilt/zoom per frame from pitch lines -> calib.pkl
   `run_calib_fill.py`— backward pass that fills mid-shot gaps
3. `positions.py`     — pitch coordinates, team split, tracking -> positions.pkl
4. `model_footage.py` — depth / phase / pressure / engaged / marking / heatmap -> model_out.pkl
5. `render.py`        — draws everything onto the video -> overlay mp4
6. `benchmark.py`     — compares against PFF tracking for the same minutes

Not included, needed to run:
- The detector weights: stock COCO YOLOv8n as ONNX, taken from the npm package
  `node-red-contrib-yolov8@0.1.0` (`lib/model/yolov8n.onnx`); path set in `detect.py`.
- Fonts: `@fontsource/barlow-condensed` and `@fontsource/ibm-plex-mono` (npm); paths in `render.py`.
- `heatmap_export.py` / `space_behind_line.py` from the main pipeline (for the heatmap).

Match-specific pieces to change for other footage: `camera_C.npy` (camera
position — re-solve with `lines_annot.py` + `solve_from_lines.py` on one clear
frame), the kit-colour rules in `positions.py` (`classify_team`), the attacking
direction in `model_footage.py` (`HOME_POSITIVE`), the live window in
`run_calib.py`/`render.py`, and the screen-overlay boxes in `fit_camera.py` (`OV`).
