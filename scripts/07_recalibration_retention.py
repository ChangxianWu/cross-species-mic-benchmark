# -*- coding: utf-8 -*-
"""信号保留比例、重标定抽样策略、删失影响与判别异质性（均在 L4 上）。

  保留比例  逐拟合配对计算 retention = r2_affine_oracle / 同单元 L1 R²，并给出
            k 标签重标定的保留比例——只报比值，不把 oracle 上界读成「信号完好」。
  抽样策略  比较三种本地重标定抽样（随机 / 按预测分位分层 / 按观测表型分层），
            每个 (species, drug) 单元独立重标定。
  删失      在 L4 上同时扰动训练与测试标签（左截尾 −0.5 / 右截尾 +0.5），
            重算「位移量」与「均值对齐 R²」，看跨物种结论是否随删失处理改变。
  判别异质性 按留出物种、按药物、按表型流行率分层报告 AUC 分布。
  尺度错配  产出每个 L4 拟合的最优斜率 b，量化「尺度错配」而非「非线性」。

输出：output/recalibration_retention.tsv
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = Path(BASE) / "data"
OUT = Path(BASE) / "output"
SUBS = DATA / "unit_subsets"
FEAT = DATA / "feature_raw.parquet"
SEED = 20261011
TARGET = "log2mic"
N_JOBS = 12
KS = (25, 50, 100)
REPS = 10

_CACHE = {}


def feats():
    if not _CACHE:
        df = pd.read_parquet(FEAT)
        df["genome_id"] = df["genome_id"].astype(str)
        _CACHE["raw"] = df.set_index("genome_id")
    return _CACHE["raw"]


def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08,
                        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                        n_jobs=2, random_state=SEED, verbosity=0)


MODELS = {
    "ridge": (lambda: Ridge(alpha=10.0), True),
    "rf": (lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                         n_jobs=2, random_state=SEED), False),
    "xgb": (_xgb, False),
}


def fit_predict(fn, scale, Xtr, ytr, Xte):
    if scale:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = fn()
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def affine(y_cal, yh_cal, yh_test):
    """在标定集上拟合 y = a + b·ŷ，返回 (b, 预测)。"""
    v = np.var(yh_cal)
    if v < 1e-12:
        return np.nan, np.full(len(yh_test), float(np.mean(y_cal)))
    b = float(np.cov(yh_cal, y_cal, ddof=0)[0, 1] / v)
    a = float(np.mean(y_cal) - b * np.mean(yh_cal))
    return b, a + b * yh_test


def perturb_censoring(vals, cens, direction):
    """direction=+1：左截尾下移、右截尾上移（区间变宽）；-1 反向。"""
    v = vals.astype(float).copy()
    if direction > 0:
        v[cens == "left"] -= 0.5
        v[cens == "right"] += 0.5
    else:
        v[cens == "left"] += 0.5
        v[cens == "right"] -= 0.5
    return v


def analyse(drug, held, tr_parquet, te_parquet, key):
    trd = pd.read_parquet(tr_parquet)
    ted = pd.read_parquet(te_parquet)
    X = feats()
    tr_ids = [str(g) for g in trd["genome_id"]]
    te_ids = [str(g) for g in ted["genome_id"]]
    Xtr, Xte = X.loc[tr_ids].to_numpy(np.float32), X.loc[te_ids].to_numpy(np.float32)
    ytr = trd[TARGET].to_numpy(float)
    yte = ted[TARGET].to_numpy(float)
    ctr = trd["cens"].fillna("exact").to_numpy()
    cte = ted["cens"].fillna("exact").to_numpy()
    yb = np.array([1 if s in ("Resistant", "Nonsusceptible") else
                   0 if s in ("Susceptible", "Susceptible-dependent") else -1
                   for s in ted["pheno"].fillna("")], dtype=int)
    prev = float((yb[yb >= 0] == 1).mean()) if (yb >= 0).sum() else np.nan
    rows = []
    rng = np.random.RandomState(SEED)
    for mname, (fn, scale) in MODELS.items():
        try:
            yh = fit_predict(fn, scale, Xtr, ytr, Xte)
        except Exception:  # noqa: BLE001
            continue
        r2_raw = r2_score(yte, yh)
        r2_mean = r2_score(yte, yh + (yte.mean() - yh.mean()))
        b, _ = affine(yte, yh, yh)          # oracle 斜率（用目标标签）
        r2_orc = r2_score(yte, np.poly1d(np.polyfit(yh, yte, 1))(yh))
        # --- N3 抽样策略 ---
        strat = {}
        for k in KS:
            if k >= len(yte) - 20:
                continue
            acc = {"random": [], "pred_strat": [], "pheno_strat": []}
            for _ in range(REPS):
                idx = rng.permutation(len(yte))
                cal, test = idx[:k], idx[k:]
                _, pred = affine(yte[cal], yh[cal], yh[test])
                acc["random"].append(r2_score(yte[test], pred))
                # 按预测值分位分层抽样（覆盖预测范围）
                q = pd.qcut(pd.Series(yh), q=min(5, len(yh) // max(k, 1) + 1),
                            labels=False, duplicates="drop")
                if q.notna().all() and q.nunique() >= 2:
                    per = max(1, k // q.nunique())
                    sel = []
                    for g in sorted(q.dropna().unique()):
                        pool = np.where(q.to_numpy() == g)[0]
                        take = min(per, len(pool))
                        sel.extend(rng.choice(pool, take, replace=False))
                    sel = np.array(sel[:k], dtype=int)
                    rest = np.setdiff1d(np.arange(len(yte)), sel)
                    if len(rest) > 20:
                        _, pred = affine(yte[sel], yh[sel], yh[rest])
                        acc["pred_strat"].append(r2_score(yte[rest], pred))
                # 按观测表型分层（若可用）
                if (yb >= 0).all() and len(set(yb)) == 2:
                    sel = []
                    for g in (0, 1):
                        pool = np.where(yb == g)[0]
                        take = min(max(1, k // 2), len(pool))
                        sel.extend(rng.choice(pool, take, replace=False))
                    sel = np.array(sel[:k], dtype=int)
                    rest = np.setdiff1d(np.arange(len(yte)), sel)
                    if len(rest) > 20:
                        _, pred = affine(yte[sel], yh[sel], yh[rest])
                        acc["pheno_strat"].append(r2_score(yte[rest], pred))
            for name, vals in acc.items():
                if vals:
                    strat[f"shot{k}_{name}"] = float(np.mean(vals))
        # --- N4 跨物种删失敏感性 ---
        y_tr_w = perturb_censoring(ytr, ctr, +1)
        y_te_w = perturb_censoring(yte, cte, +1)
        y_tr_n = perturb_censoring(ytr, ctr, -1)
        y_te_n = perturb_censoring(yte, cte, -1)
        try:
            yh_w = fit_predict(fn, scale, Xtr, y_tr_w, Xte)
            r2_mean_w = r2_score(y_te_w, yh_w + (y_te_w.mean() - yh_w.mean()))
            shift_w = float(np.mean(y_tr_w) - np.mean(y_te_w))
        except Exception:  # noqa: BLE001
            r2_mean_w, shift_w = np.nan, np.nan
        try:
            yh_n = fit_predict(fn, scale, Xtr, y_tr_n, Xte)
            r2_mean_n = r2_score(y_te_n, yh_n + (y_te_n.mean() - yh_n.mean()))
            shift_n = float(np.mean(y_tr_n) - np.mean(y_te_n))
        except Exception:  # noqa: BLE001
            r2_mean_n, shift_n = np.nan, np.nan
        auc = np.nan
        if (yb >= 0).sum() >= 30 and len(set(yb[yb >= 0])) == 2:
            try:
                auc = roc_auc_score(yb[yb >= 0], yh[yb >= 0])
            except Exception:  # noqa: BLE001
                auc = np.nan
        rows.append(dict(drug=drug, species=held, model=mname,
                         n_train=len(trd), n_test=len(ted),
                         n_train_species=trd["species"].nunique(),
                         r2_raw=r2_raw, r2_mean=r2_mean, r2_affine=r2_orc,
                         slope_b=b, spearman=(spearmanr(yte, yh).correlation
                                              if np.std(yh) > 1e-9 else np.nan),
                         cens_left_frac=float((cte == "left").mean()),
                         cens_right_frac=float((cte == "right").mean()),
                         shift=float(ytr.mean() - yte.mean()),
                         r2_mean_censor_widen=r2_mean_w,
                         shift_censor_widen=shift_w,
                         r2_mean_censor_narrow=r2_mean_n,
                         shift_censor_narrow=shift_n,
                         auc=auc, prev=prev, **strat))
    return rows


def main():
    lab = pd.read_parquet(DATA / "labels_mic.parquet")
    lab = lab[lab["genotype_hit"] == 1].dropna(subset=[TARGET]).copy()
    shared = (lab.groupby(["antibiotic", "species"])["genome_id"].nunique()
              .reset_index().rename(columns={"genome_id": "g"}))
    ok = [d for d, grp in shared.groupby("antibiotic") if (grp["g"] >= 100).sum() >= 2]
    # 同单元 L1 R²（用于 retention 分母）
    s4 = pd.read_csv(OUT / "results_all.tsv", sep="\t")
    s4 = s4[s4["err"].isna()] if "err" in s4 else s4
    l1 = (s4[(s4["level"] == "L1") & (s4["feature"] == "raw")]
          .groupby(["species", "antibiotic", "model"])["r2"].mean().rename("r2_L1"))
    tasks = []
    for drug in ok:
        sds = shared[(shared["antibiotic"] == drug) & (shared["g"] >= 100)]["species"].tolist()
        for held in sds:
            k = f"{drug.replace('/', '_').replace(' ', '')}__{held.replace(' ', '_')}"
            tj, ej = SUBS / f"_tr_{k}.parquet", SUBS / f"_te_{k}.parquet"
            if tj.exists() and ej.exists():
                tasks.append((drug, held, str(tj), str(ej), k))
    print(f"[1] L4 任务 {len(tasks)}", flush=True)
    rows = Parallel(n_jobs=N_JOBS, verbose=2)(delayed(analyse)(*t) for t in tasks)
    d = pd.DataFrame([r for sub in rows for r in sub])
    d = d.merge(l1.reset_index().rename(columns={"antibiotic": "drug"}),
                on=["species", "drug", "model"], how="left")
    d["retention_oracle"] = d["r2_affine"] / d["r2_L1"]
    d["retention_shot100"] = d.get("shot100_random") / d["r2_L1"]
    d.to_csv(OUT / "recalibration_retention.tsv", sep="\t", index=False, encoding="utf-8-sig")

    print("\n=== N1 信号保留比例（相对同单元 L1）===")
    print(d[["r2_L1", "r2_affine", "r2_mean", "retention_oracle", "retention_shot100"]]
          .describe().loc[["mean", "50%"]].round(3).to_string())
    print("\n=== 最优斜率 b 的分布（b≠1 即尺度错配）===")
    print(d["slope_b"].describe()[["count", "mean", "25%", "50%", "75%"]].round(3).to_string())
    print(f"  |b−1|>0.5 的拟合比例: {(d['slope_b'].sub(1).abs() > 0.5).mean():.3f}")
    print("\n=== 抽样策略对比 ===")
    for k in KS:
        cols = [c for c in d.columns if c.startswith(f"shot{k}_")]
        if cols:
            print(f"  k={k}: " + "  ".join(f"{c.replace(f'shot{k}_',''):11s}={d[c].mean():7.3f}"
                                           for c in cols))
    print("\n=== 跨物种删失敏感性 ===")
    print(f"  位移(原)={d['shift'].abs().mean():.3f}  变宽={d['shift_censor_widen'].abs().mean():.3f}"
          f"  变窄={d['shift_censor_narrow'].abs().mean():.3f}")
    print(f"  均值对齐R²  原={d['r2_mean'].mean():.3f}  变宽={d['r2_mean_censor_widen'].mean():.3f}"
          f"  变窄={d['r2_mean_censor_narrow'].mean():.3f}")
    print("\n=== AUC 异质性 ===")
    a = d.dropna(subset=["auc"])
    print(f"  总 n={len(a)}  mean={a['auc'].mean():.3f}  IQR=[{a['auc'].quantile(.25):.3f},"
          f"{a['auc'].quantile(.75):.3f}]")
    print("  按留出物种:")
    print(a.groupby("species")["auc"].agg(["size", "mean", "median"]).round(3).to_string())
    print("  按药物（n>=8）:")
    g = a.groupby("drug")["auc"].agg(["size", "mean"])
    print(g[g["size"] >= 8].sort_values("mean").round(3).to_string())
    print("\n完成")


if __name__ == "__main__":
    main()
