# -*- coding: utf-8 -*-
"""
select_lightqdrl_x.py — pre-registered choice of the recommended configuration LightQDRL-X (validation seeds only).

Rule (fixed before the test seeds are analysed; write it in the Methods):
  1. Use only validation runs (exp_qrl_bench.py --seed_offset 100 -> results_qrl_validation/mec/per_seed.csv).
  2. Admissible algorithms: the compressed (light) agent does not lose more than delta (default 0.05 normalized units)
     against its own uncompressed agent on the validation seeds (mean paired difference >= -delta).
  3. Among admissible algorithms choose the one with the highest mean light normalized score.
  4. Ties (|difference| < tie, default 0.01) are broken by the lower mean per-update latency of the light agent.
  If no algorithm is admissible, the light agent with the highest score is chosen and the file says so.
Output: results_qrl_validation/selection.json  (rule, table of all candidates, chosen algorithm)
Usage:  python select_lightqdrl_x.py
"""
import os
import json
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", default="results_qrl_validation/mec/per_seed.csv")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--tie", type=float, default=0.01)
    ap.add_argument("--out", default="results_qrl_validation/selection.json")
    a = ap.parse_args()
    df = pd.read_csv(a.val_csv)
    df = df[df.core == "vqc"]
    rows = []
    for al, d in df.groupby("algo"):
        p = d.pivot_table(index="seed", columns="variant", values="norm_score").dropna()
        lat = d[d.variant == "light"]["update_ms_mean"].mean()
        rows.append(dict(algo=al, n_seeds=len(p), base=float(p["base"].mean()), light=float(p["light"].mean()),
                         diff=float((p["light"] - p["base"]).mean()), light_update_ms=float(lat)))
    t = pd.DataFrame(rows)
    t["admissible"] = t["diff"] >= -a.delta
    cand = t[t.admissible] if t.admissible.any() else t
    best = cand.light.max()
    tied = cand[cand.light >= best - a.tie].sort_values("light_update_ms")
    chosen = tied.iloc[0].algo
    res = dict(rule=dict(delta=a.delta, tie=a.tie, text=__doc__.split("Rule")[1].split("Output")[0].strip()),
               any_admissible=bool(t.admissible.any()), chosen=chosen,
               candidates=t.sort_values("light", ascending=False).to_dict(orient="records"))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(t.sort_values("light", ascending=False).round(3).to_string(index=False))
    print("LightQDRL-X =", chosen, "" if res["any_admissible"] else "(no algorithm admissible: chosen by score only)")


if __name__ == "__main__":
    main()
