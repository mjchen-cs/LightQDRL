# -*- coding: utf-8 -*-
"""
make_data_package.py -- build the data deposit for the LightQDRL paper (Zenodo / Figshare).

Run from the LightQDRL folder:
    python make_data_package.py
Optional:  --doi 10.5281/zenodo.XXXXXXX   (writes the DOI into README.md and the citation file)
           --out LightQDRL_data_v1.zip

The script copies nothing and deletes nothing in the project folder.  It writes one zip archive that contains
  README.md            what is in the deposit and which figure / table each part supports
  MANIFEST.csv         every file with its size and SHA-256 checksum
  LICENSE.txt          CC BY 4.0
  .zenodo.json         metadata that Zenodo reads when the archive is uploaded through GitHub (optional)
  results/...          the data (see FOLDERS below)
Superseded runs (scalar-reward 'unfactored' runs, failed Quafu submissions, __pycache__) are excluded.
"""
import os
import io
import re
import csv
import sys
import json
import glob
import time
import hashlib
import zipfile
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))

# (source path relative to the LightQDRL folder, path inside the archive)
FOLDERS = [
    ("results_qrl/mec", "results/benchmark/wp_mec"),
    ("results_qrl/channel", "results/benchmark/multichannel_access"),
    ("results_qrl/cartpole", "results/benchmark/cartpole"),
    ("results_qrl_validation", "results/selection_validation_seeds"),
    ("results_qrl_adapt/mec", "results/adaptation/wp_mec"),
    ("results_qrl_robust/mec", "results/robustness/wp_mec"),
    ("edge_results_jetson", "results/on_device_jetson_orin_nano_super"),
    ("results_hw/Q-PPO_seed0_WK_C180_s1000", "results/hardware_origin_wk_c180/run_Q-PPO_seed0_1000shots"),
]
SINGLE_FILES = [
    ("results_hw/chip_WK_C180.json", "results/hardware_origin_wk_c180/chip_calibration_snapshot_WK_C180.json"),
]
# checkpoints of the agent executed on hardware (trained in the earlier WP-MEC version, see paper)
GLOBS = [
    ("results_qrl_v1/mec/runs/Q-PPO_base_vqc_seed0.*", "results/hardware_origin_wk_c180/agent_checkpoints_env_v1"),
    ("results_qrl_v1/mec/runs/Q-PPO_light_vqc_seed0.*", "results/hardware_origin_wk_c180/agent_checkpoints_env_v1"),
    ("results_qrl_v1/mec/reference_policies.json", "results/hardware_origin_wk_c180/agent_checkpoints_env_v1"),
]
EXCLUDE = re.compile(r"(unfactored|__pycache__|\.ipynb_checkpoints|\.DS_Store|Thumbs\.db|~\$)", re.I)

README = r"""# Data for "Lightweight hybrid quantum-classical decision policies for resource-constrained edge computing"

Miaojiang Chen, Wenjing Xiao, Haiqiang Chen, Zhenrong Zhang, Guihua Luo, Athanasios Vasilakos, Ahmed Farouk

DOI: {doi}
Code: https://github.com/mjchen-cs/LightQDRL
License: CC BY 4.0 (see LICENSE.txt)

This archive contains the per-seed results, run configurations, trained checkpoints, adaptation and robustness
records, on-device measurements and raw hardware counts (with the chip calibration snapshot) that underlie every
figure and table of the paper and its Supplementary Information. Every file is listed with its SHA-256 checksum in
MANIFEST.csv. Archive built on {built} from {nfiles} files ({mb:.1f} MB uncompressed).

## Contents and where each part is used

| Folder | Content | Paper |
|---|---|---|
| results/benchmark/wp_mec | Per-seed results (per_seed.csv), reference policies, run configuration, simulated latency (latency.csv) and, for every run, the checkpoint (.pt), run record (.json) and learning curves (.npz). Algorithms Q-DQN, Q-DDQN, Q-DuelingDQN, Q-A2C, Q-A3C, Q-PPO, Q-TRPO; variants base, light, dqcp, jcqq; cores vqc (hybrid), tanh (quantum removal), mlp (classical); seeds 0-9 | Figs. 3-5; Tables 1-2; Supplementary Tables 1-3; Supplementary Note 2 |
| results/benchmark/multichannel_access | Same for the multichannel-access task | Fig. 3; Table 1; Supplementary Table 1 |
| results/benchmark/cartpole | Same for CartPole-v1 | Fig. 3; Table 1; Supplementary Table 1 |
| results/selection_validation_seeds | Runs on validation seeds 100-102 and the pre-registered selection (selection.json) | Fig. 5a |
| results/adaptation/wp_mec | Adaptation of all WP-MEC agents to the traffic and channel shifts (full fine-tuning, PEQA, frozen); per_run.csv and per-run curves; shifted-system reference policies | Figs. 5c-e and 6 |
| results/robustness/wp_mec | Decision agreement under gate noise, finite shots and routing, and closed-loop scores with noisy circuits; per_model.csv, per-model records and the fixed probe states | Fig. 7a-e |
| results/on_device_jetson_orin_nano_super | edge_latency.csv (latency, stored size, decision agreement, closed-loop score, power and energy for every model and format) and device_info.json | Table 3; Supplementary Tables 6-7; Supplementary Fig. 1 |
| results/hardware_origin_wk_c180 | Raw counts of both circuits on 70 states (raw_counts_*.json), per-state results, bit-order calibration, job identifiers, Qiskit compilation estimate, the chip calibration snapshot, and the checkpoints of the executed agent | Fig. 7f; Supplementary Tables 4 and 8; Supplementary Fig. 2 |
| inputs/video_task_sizes.csv | Sizes in bytes of the 58 industrial video clips that define the task-size distribution of WP-MEC (anonymized clip identifiers; not in MANIFEST.csv because it is generated during packaging) | Fig. 2b |

## Notes

* Normalized scores: (R - R_random) / (R_reference - R_random) with the myopic max-weight (WP-MEC) or belief-myopic
  (multichannel access) reference stored in reference_policies.json; CartPole: R / 500.
* Test seeds 0-9 and validation seeds 100-102 are disjoint; base and light agents of a pair share the seed.
* Checkpoints (.pt) are PyTorch dictionaries (state, algo, variant, core, obs_dim, branches, heads, compress, env,
  hp, critic) and are loaded with the code repository (qrl_core.HybridNet).
* The hardware run used one Q-PPO agent trained in an earlier version of the WP-MEC environment with the identical
  circuit architecture (make_env("mec_v1") in the code); its checkpoints are included.
* Mean channel gains are taken from the public dataset of the DROO framework (Huang et al., IEEE TMC 2020) and are
  distributed with the code repository, not with this archive.
* Runs obtained with the superseded scalar-reward version of WP-MEC (Supplementary Note 1) are not included.
"""

ZENODO = {
    "title": "Data for: Lightweight hybrid quantum-classical decision policies for resource-constrained edge computing",
    "upload_type": "dataset",
    "license": "cc-by-4.0",
    "creators": [{"name": n} for n in ["Chen, Miaojiang", "Xiao, Wenjing", "Chen, Haiqiang", "Zhang, Zhenrong",
                                       "Luo, Guihua", "Vasilakos, Athanasios", "Farouk, Ahmed"]],
    "keywords": ["lightweight machine learning", "edge computing", "hybrid quantum-classical",
                 "reinforcement learning", "model compression", "quantization", "Jetson", "superconducting processor"],
    "related_identifiers": [{"identifier": "https://github.com/mjchen-cs/LightQDRL", "relation": "isSupplementedBy"}],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect():
    items, missing = [], []
    for src, dst in FOLDERS:
        base = os.path.join(ROOT, src)
        if not os.path.isdir(base):
            missing.append(src)
            continue
        for d, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs if not EXCLUDE.search(x)]
            for fn in files:
                p = os.path.join(d, fn)
                rel = os.path.relpath(p, base).replace(os.sep, "/")
                if EXCLUDE.search(rel):
                    continue
                items.append((p, f"{dst}/{rel}"))
    for src, dst in SINGLE_FILES:
        p = os.path.join(ROOT, src)
        (items.append((p, dst)) if os.path.isfile(p) else missing.append(src))
    for pattern, dst in GLOBS:
        hits = glob.glob(os.path.join(ROOT, pattern))
        if not hits:
            missing.append(pattern)
        for p in hits:
            items.append((p, f"{dst}/{os.path.basename(p)}"))
    seen, uniq = set(), []
    for p, a in items:
        if a not in seen:
            seen.add(a)
            uniq.append((p, a))
    return sorted(uniq, key=lambda t: t[1]), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="LightQDRL_data_v1.zip")
    ap.add_argument("--doi", default="(assigned by the repository on publication)")
    a = ap.parse_args()
    items, missing = collect()
    if missing:
        print("WARNING - not found (skipped):")
        for m in missing:
            print("   ", m)
    if not items:
        sys.exit("nothing to package - run this script inside the LightQDRL folder")
    rows, total = [], 0
    for i, (p, arc) in enumerate(items, 1):
        n = os.path.getsize(p)
        total += n
        rows.append((arc, n, sha256(p)))
        if i % 500 == 0:
            print(f"  hashed {i}/{len(items)} files", flush=True)
    out = os.path.join(ROOT, a.out)
    built = time.strftime("%Y-%m-%d")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p, arc in items:
            z.write(p, arc)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["path", "bytes", "sha256"])
        w.writerows(rows)
        z.writestr("MANIFEST.csv", buf.getvalue())
        z.writestr("README.md", README.format(doi=a.doi, built=built, nfiles=len(items), mb=total / 1e6))
        z.writestr("LICENSE.txt", "This dataset is licensed under the Creative Commons Attribution 4.0 International "
                                  "License (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/\n")
        z.writestr(".zenodo.json", json.dumps(ZENODO, indent=2, ensure_ascii=False))
        # clip sizes, anonymized: original file names (site folders, timestamps, descriptions) are not distributed
        vf = os.path.join(ROOT, "data", "video_task_sizes.csv")
        if os.path.isfile(vf):
            with open(vf, "r", encoding="utf-8-sig", newline="") as fh:
                sizes = [r[-1] for r in list(csv.reader(fh))[1:] if r and r[-1].strip()]
            vb = io.StringIO()
            vw = csv.writer(vb, lineterminator="\n")
            vw.writerow(["clip_id", "bytes"])
            vw.writerows([(f"clip{i + 1:02d}", s) for i, s in enumerate(sizes)])
            z.writestr("inputs/video_task_sizes.csv", vb.getvalue())
            print(f"  inputs/video_task_sizes.csv: {len(sizes)} clip sizes (file names removed)")
        else:
            print("WARNING - data/video_task_sizes.csv not found")
    print(f"\nwrote {out}")
    print(f"  {len(items)} data files, {total / 1e6:.1f} MB uncompressed, archive {os.path.getsize(out) / 1e6:.1f} MB")
    by_top = {}
    for arc, n, _ in rows:
        k = "/".join(arc.split("/")[:2])
        by_top[k] = by_top.get(k, 0) + n
    for k, n in sorted(by_top.items()):
        print(f"  {k:55s} {n / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
