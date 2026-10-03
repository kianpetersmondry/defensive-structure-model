"""Assembles pressure_read.html from the head/tail templates plus the two
large JSON data blobs (match tracking data, heatmap raster data), without
ever holding both source strings AND a second copy in memory at once via
naive string concatenation of huge literals -- streamed straight to disk.

artifact_head.html ends with the OPENING <script id="match-data" ...> tag.
We close it after animation_data.json, open a second <script id="heatmap-data">
tag, drop in heatmap_data.json, and let artifact_tail.html's own leading
</script> close that second tag before the main IIFE starts -- same
structural trick the head/tail split already used for one blob, just
inserted twice.
"""
HEAD = '/home/claude/project_work/artifact_head.html'
ANIM = '/home/claude/project_work/animation_data.json'
HEAT = '/home/claude/project_work/heatmap_data.json'
TAIL = '/home/claude/project_work/artifact_tail.html'
OUT = '/home/claude/project_work/pressure_read.html'

with open(OUT, 'wb') as out:
    with open(HEAD, 'rb') as f:
        out.write(f.read())
    with open(ANIM, 'rb') as f:
        out.write(f.read())
    out.write(b'\n</script>\n<script id="heatmap-data" type="application/json">\n')
    with open(HEAT, 'rb') as f:
        out.write(f.read())
    with open(TAIL, 'rb') as f:
        out.write(f.read())

import os
print(f"Wrote {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB)")
