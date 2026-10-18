# -*- coding: utf-8 -*-
"""Rebuild data/video_task_sizes.csv from a folder of video clips (default: './task data').

The released file lists anonymized clip identifiers (clip01, clip02, ...) and sizes in bytes, in the order of a
sorted directory walk; qrl_envs.py only uses the sizes.  Usage: python make_video_sizes.py ["path/to/videos"]"""
import os, sys, csv
root = sys.argv[1] if len(sys.argv) > 1 else "task data"
sizes = []
for d, dirs, files in os.walk(root):
    dirs.sort()
    for f in sorted(files):
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            sizes.append(os.path.getsize(os.path.join(d, f)))
os.makedirs("data", exist_ok=True)
with open(os.path.join("data", "video_task_sizes.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["clip_id", "bytes"])
    w.writerows((f"clip{i + 1:02d}", s) for i, s in enumerate(sizes))
print(f"{len(sizes)} clip sizes written to data/video_task_sizes.csv")
