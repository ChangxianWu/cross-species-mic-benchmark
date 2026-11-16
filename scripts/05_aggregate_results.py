# -*- coding: utf-8 -*-
"""从断点单元重建结果表（不重算任何模型，耗时秒级）。

04 的 collect() 依赖统一的断点文件扩展名；一旦扩展名不一致就会漏收，
导致 results_all.tsv 只剩 L4。此处按「读断点目录下全部文件、按内容解析」
重新汇总，因此也顺手挡住目录项与隐藏文件被当成结果行。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = Path(BASE) / "data"
OUT = Path(BASE) / "output"
PARTS = Path(BASE) / "cache" / "calibration_parts"   # 与 04 同源（断点也在 cache/）
ALPHA = 0.10


def collect(prefix=""):
    rows = []
    for p in sorted(PARTS.glob("*")):
        # 只收顶层 JSON 文件：跳过目录与工具状态目录（.mimosa 里也有 .json，
        # 会被当成结果行并炸掉列结构 —— 2026-10-11 实测）
        if not p.is_file() or p.suffix != ".json" or p.name.startswith("."):
            continue
        name = p.stem
        if name.startswith("DENS_"):
            if prefix != "DENS_":
                continue
        elif prefix and not name.startswith(prefix):
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(rec, list):
            raise RuntimeError(f"[FATAL] 断点文件内容不是数组：{p}")
        rows.extend(rec)
    return pd.DataFrame(rows)


def main():
    allr = collect()
    n_err = int((allr.get("err", pd.Series(dtype=str)) != "").sum()) if "err" in allr else 0
    allr = allr[allr["err"] == ""] if "err" in allr else allr
    allr.to_csv(OUT / "results_all.tsv", sep="\t", index=False, encoding="utf-8-sig")
    print(f"all_results 行={len(allr)}（丢弃 err 行 {n_err}）；"
          f"level 分布 {allr['level'].value_counts().to_dict()}")

    cal = (allr.groupby(["feature", "model", "level"])
           .agg(n=("r2", "size"), r2=("r2", "mean"), r2_shift=("r2_shift", "mean"),
                cal_slope=("cal_slope", "mean"), cal_slope_sd=("cal_slope", "std"),
                cal_intercept=("cal_intercept", "mean"),
                cal_e50=("cal_e50", "mean"), cal_e90=("cal_e90", "mean"),
                cal_bin_mae=("cal_bin_mae", "mean")).reset_index())
    cal.to_csv(OUT / "calibration.tsv", sep="\t", index=False, encoding="utf-8-sig")

    conf = (allr.groupby(["feature", "model", "level"])
            .agg(n=("cov", "size"), cov=("cov", "mean"), cov_sd=("cov", "std"),
                 width=("width", "mean"), cov_eff=("cov_eff", "mean"),
                 rmse=("rmse", "mean")).reset_index())
    conf["target_cov"] = 1 - ALPHA
    conf.to_csv(OUT / "conformal.tsv", sep="\t", index=False, encoding="utf-8-sig")

    abl = (allr.groupby(["feature", "model", "level"])
           .agg(r2=("r2", "mean"), r2_shift=("r2_shift", "mean"),
                rmse=("rmse", "mean"), mae=("mae", "mean")).reset_index())
    abl.to_csv(OUT / "ablation.tsv", sep="\t", index=False, encoding="utf-8-sig")

    d = collect("DENS_")
    if len(d):
        d.to_csv(OUT / "density_decomposition.tsv", sep="\t", index=False,
                 encoding="utf-8-sig")

    print("\n=== R² 均值（含 L1–L4）===")
    print(abl.pivot_table(index=["feature", "model"], columns="level",
                          values="r2").round(3).to_string())
    print("\n=== 偏置校正 R²（r2_shift，隔离均值位移后）===")
    print(abl.pivot_table(index=["feature", "model"], columns="level",
                          values="r2_shift").round(3).to_string())
    print("\n=== 校准斜率 ===")
    print(cal.pivot_table(index=["feature", "model"], columns="level",
                          values="cal_slope").round(3).to_string())
    print("\n=== Conformal 覆盖率（目标 0.90）===")
    print(conf.pivot_table(index=["feature", "model"], columns="level",
                           values="cov").round(3).to_string())
    print("\n=== Conformal 平均区间宽度 ===")
    print(conf.pivot_table(index=["feature", "model"], columns="level",
                           values="width").round(3).to_string())


if __name__ == "__main__":
    main()
