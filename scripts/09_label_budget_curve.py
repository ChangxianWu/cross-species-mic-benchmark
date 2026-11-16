# -*- coding: utf-8 -*-
"""标签预算曲线：迁移 + 重标定 vs 本地训练（交叉验证口径）。

要点：同物种「上界」必须在**训练折**内拟合、**留出折**上评估；在评估子集上训练
得到的数字是训练集内表现，不能当上界。三条路径共用同一折划分，严格可比。

  A 超参一致性  两条路径用同一个模型构造器、同一组固定超参；另在 k=100 处加一组
                「两边都允许内层 CV 调参」的对照，验证结论不是调参不对称造成的。
  B 预算曲线    k ∈ {25,50,100,200,400} 给出迁移 vs 本地的完整曲线，并求每个单元的
                转折点（本地开始稳定胜出的最小 k）。
  C 适用面      筛查兜底的适用面：conc≥0.6 的单元占比，以及该任务下耐药流行率。
  D 过拟合风险  本地训练在高维特征下的风险：比较固定强正则（alpha=10）与内层
                CV 选 alpha。

输出：output/label_budget.tsv
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
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
KGRID = (25, 50, 100, 200, 400)
FOLDS = 3
REPS = 3

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


# 迁移与本地共用同一构造器与超参（保证两者对称）
MODELS = {
    "ridge": lambda: Ridge(alpha=10.0),
    "rf": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                        n_jobs=2, random_state=SEED),
    "xgb": _xgb,
}
NEEDS_SCALE = {"ridge": True, "rf": False, "xgb": False}


def fit_pred(mname, Xtr, ytr, Xte, tune=False):
    Xtr, Xte = np.asarray(Xtr), np.asarray(Xte)
    if NEEDS_SCALE[mname]:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if tune and mname == "ridge":
        # 允许内层 CV 选正则强度（检验「本地训练被固定强正则拖累」这一可能性）
        m = RidgeCV(alphas=(0.1, 1, 10, 100, 1000, 10000), cv=3).fit(Xtr, ytr)
    else:
        m = MODELS[mname]().fit(Xtr, ytr)
    return m.predict(Xte)


def recal(y_cal, yh_cal, yh_out):
    v = np.var(yh_cal)
    if v < 1e-12:
        return np.full(len(yh_out), float(np.mean(y_cal)))
    b = float(np.cov(yh_cal, y_cal, ddof=0)[0, 1] / v)
    return float(np.mean(y_cal) - b * np.mean(yh_cal)) + b * np.asarray(yh_out)


def analyse(drug, held, tr_parquet, te_parquet, key):
    trd = pd.read_parquet(tr_parquet)
    ted = pd.read_parquet(te_parquet).sort_values("val").drop_duplicates("genome_id",
                                                                       keep="first")
    X = feats()
    tr_ids = [str(g) for g in trd["genome_id"]]
    te_ids = [str(g) for g in ted["genome_id"]]
    Xtr = X.loc[tr_ids].to_numpy(np.float32)
    Xte = X.loc[te_ids].to_numpy(np.float32)
    ytr = trd[TARGET].to_numpy(float)
    yte = ted[TARGET].to_numpy(float)
    n = len(yte)
    if n < 80:
        return []
    rows = []
    rng = np.random.RandomState(SEED)
    for mname in MODELS:
        # 迁移模型：一次拟合，各 k 的重标定只花 O(k)
        try:
            yh_tr = fit_pred(mname, Xtr, ytr, Xte)
        except Exception:  # noqa: BLE001
            continue
        acc = {f"transfer_k{k}": [] for k in KGRID}
        acc.update({f"local_k{k}": [] for k in KGRID})
        acc.update({f"local_tuned_k{k}": [] for k in KGRID})
        fold_ceiling = []
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        # KFold.split 产出 (train_idx, test_idx) 元组——直接解包，
        # 误把元组当 test_idx 会让 setdiff1d 收到形状 (2,) 的对象数组而报
        # "inhomogeneous shape"
        for _tr_idx, test_idx in kf.split(np.arange(n)):
            pool = np.setdiff1d(np.arange(n), test_idx)
            # 同物种上界：只用训练折（不含测试折）拟合，评估留出折
            try:
                yh_ceiling = fit_pred(mname, Xte[pool], yte[pool], Xte[test_idx])
                fold_ceiling.append(r2_score(yte[test_idx], yh_ceiling))
            except Exception:  # noqa: BLE001
                pass
            for _ in range(REPS):
                perm = rng.permutation(pool)
                for k in KGRID:
                    if k >= len(pool) - 10:
                        continue
                    cal = perm[:k]
                    acc[f"transfer_k{k}"].append(
                        r2_score(yte[test_idx], recal(yte[cal], yh_tr[cal],
                                                      yh_tr[test_idx])))
                    try:
                        acc[f"local_k{k}"].append(
                            r2_score(yte[test_idx],
                                     fit_pred(mname, Xte[cal], yte[cal], Xte[test_idx])))
                    except Exception:  # noqa: BLE001
                        pass
                    if k == 100 and mname == "ridge":
                        acc[f"local_tuned_k{k}"].append(
                            r2_score(yte[test_idx],
                                     fit_pred(mname, Xte[cal], yte[cal], Xte[test_idx],
                                              tune=True)))
        row = dict(drug=drug, species=held, model=mname, n_test=n,
                   n_features=int(Xtr.shape[1]),
                   within_cv=float(np.mean(fold_ceiling)) if fold_ceiling else np.nan)
        for k in KGRID:
            for p in ("transfer", "local", "local_tuned"):
                v = acc.get(f"{p}_k{k}")
                # v 可能是空列表（k 超过该单元测试集可用量）或含 None；
                # np.mean([]) 会产生 NaN 数组，进而让 DataFrame 出现 inhomogeneous 形状
                vals = [x for x in (v or []) if x is not None and np.isfinite(x)]
                row[f"{p}_k{k}"] = float(np.mean(vals)) if vals else np.nan
        rows.append(row)
    return rows


def main():
    lab = pd.read_parquet(DATA / "labels_mic.parquet")
    lab = lab[lab["genotype_hit"] == 1].dropna(subset=[TARGET]).copy()
    shared = (lab.groupby(["antibiotic", "species"])["genome_id"].nunique()
              .reset_index().rename(columns={"genome_id": "g"}))
    ok = [d for d, grp in shared.groupby("antibiotic") if (grp["g"] >= 100).sum() >= 2]
    tasks = []
    for drug in ok:
        sds = shared[(shared["antibiotic"] == drug) & (shared["g"] >= 100)]["species"].tolist()
        for held in sds:
            k = f"{drug.replace('/', '_').replace(' ', '')}__{held.replace(' ', '_')}"
            tj, ej = SUBS / f"_tr_{k}.parquet", SUBS / f"_te_{k}.parquet"
            if tj.exists() and ej.exists():
                tasks.append((drug, held, str(tj), str(ej), k))
    print(f"[1] L4 任务 {len(tasks)}（K 折 CV，k 网格 {KGRID}）", flush=True)
    rows = Parallel(n_jobs=N_JOBS, verbose=2)(delayed(analyse)(*t) for t in tasks)
    d = pd.DataFrame([r for sub in rows for r in sub])
    d.to_csv(OUT / "label_budget.tsv", sep="\t", index=False, encoding="utf-8-sig")

    print("\n=== 三条路径随 k 的曲线（中位 R²）===")
    hdr = f"{'path':10s}" + "".join(f"{('k=' + str(k)):>9}" for k in KGRID)
    print(hdr)
    for p in ("transfer", "local", "local_tuned"):
        vals = [d.get(f"{p}_k{k}") for k in KGRID]
        if any(v is not None and v.notna().any() for v in vals):
            print(f"{p:10s}" + "".join(
                f"{(v.median() if v is not None and v.notna().any() else float('nan')):>9.3f}"
                for v in vals))
    print(f"{'within_cv':10s}{d['within_cv'].median():>9.3f}   (交叉验证上界，与 k 无关)")

    print("\n=== 转折点：本地首次且此后均优于迁移的最小 k ===")
    cross = []
    for i, r in d.iterrows():
        ks = [k for k in KGRID if pd.notna(r.get(f"local_k{k}"))
              and pd.notna(r.get(f"transfer_k{k}"))]
        win = [k for k in ks if r[f"local_k{k}"] > r[f"transfer_k{k}"]]
        for k in ks:
            if all(r[f"local_k{kk}"] > r[f"transfer_k{kk}"] for kk in ks if kk >= k):
                cross.append(k)
                break
    if cross:
        s = pd.Series(cross)
        print(f"  可判定的拟合 {len(s)}/{len(d)}；转折点中位 k={s.median():.0f}"
              f"（IQR {s.quantile(.25):.0f}–{s.quantile(.75):.0f}）")
        print("  分布:", s.value_counts().sort_index().to_dict())
    print(f"  k=25 时本地已胜出的比例: "
          f"{(d['local_k25'] > d['transfer_k25']).mean():.3f}")

    print("\n=== 超参是否对称（ridge, k=100）===")
    r = d[d.model == "ridge"]
    print(f"  固定 alpha=10：transfer {r['transfer_k100'].median():.3f} vs "
          f"local {r['local_k100'].median():.3f}")
    if r["local_tuned_k100"].notna().any():
        print(f"  本地改用内层 CV 选 alpha：local {r['local_tuned_k100'].median():.3f} "
              f"(Δ vs 固定 {r['local_tuned_k100'].median() - r['local_k100'].median():+.3f})")
        print(f"  两种本地设定下本地仍胜出的比例："
              f"固定 {(r['local_k100'] > r['transfer_k100']).mean():.3f} / "
              f"调参 {(r['local_tuned_k100'] > r['transfer_k100']).mean():.3f}")
    print(f"  特征维度中位: {d['n_features'].median():.0f}；k=100 时的样本/特征比 "
          f"{100 / d['n_features'].median():.3f}")

    print("\n=== 筛查兜底的适用面 ===")
    # concordance 在 ordering_screening 表（recalibration_retention 无该列）
    t4 = pd.read_csv(OUT / "ordering_screening.tsv", sep="\t")
    c = t4["conc"].dropna()
    print(f"  conc 可评估拟合 {len(c)}；conc≥0.6 的占比 {(c >= 0.6).mean():.3f}"
          f"；conc≥0.7 的占比 {(c >= 0.7).mean():.3f}")
    print(f"  conc 中位 {c.median():.3f}；四分位 {c.quantile(.25):.3f}–{c.quantile(.75):.3f}")
    print("\n完成")


if __name__ == "__main__":
    main()
