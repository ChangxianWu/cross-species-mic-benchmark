# -*- coding: utf-8 -*-
"""校正阶梯：同一批跨物种预测在四种尺度假设下的表现。

要点：**强制方差匹配**（b = sd_y / sd_ŷ）不是 OLS 最优仿射重标定，会把 R² 算得
偏低；任何线性重标定能达到的最好 R² 等于 Pearson 相关系数的平方。这里给出：

  (a) r2_raw            模型原样输出（同时承载均值位移 + 尺度误差）
  (b) r2_mean           仅对齐均值
  (c) r2_affine_oracle  在测试集自身拟合 y = a + b·ŷ 后的 R²（= ρ²，上限）
  (d) r2_shot{k}        在目标物种 k 个已标注菌株上拟合 affine，其余菌株评估
                        （k = 10/25/50/100）——临床可执行的情形：
                        「一个新物种只要拿几十个菌株做本地重标定，能到多少」
  (e) auc               折点判别 AUC（阈值无关）
  (f) conc              一致性指数（连续结局的排序判别）

同时处理 sd_ratio 在零方差单元上的 inf。

输出：output/correction_ladder.tsv / correction_ladder_by_train_species.tsv
"""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau, pearsonr, spearmanr
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
SEED = 20261011
MIN_UNIT = 100
TARGET = "log2mic"
N_JOBS = 12
FEAT_RAW = DATA / "feature_raw.parquet"
K_LIST = (10, 25, 50, 100)

_CACHE = {}


def feats():
    if not _CACHE:
        df = pd.read_parquet(FEAT_RAW)
        df["genome_id"] = df["genome_id"].astype(str)
        _CACHE["raw"] = df.set_index("genome_id")
    return _CACHE["raw"]


def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08,
                        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                        n_jobs=2, random_state=SEED, verbosity=0)


def models():
    return {
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


def concordance(y, yh):
    """一致性指数（Harrell's C）：可比较对中 ŷ 与 y 同序的比例。0.5 = 无信息。"""
    n = len(y)
    num = den = 0
    for i in range(n - 1):
        dy = y[i + 1:] - y[i]
        dh = yh[i + 1:] - yh[i]
        m = dy != 0
        den += int(m.sum())
        num += int(((dy[m] > 0) == (dh[m] > 0)).sum())
    return num / den if den else np.nan


def analyse(drug, held, tr_parquet, te_parquet, key):
    trd = pd.read_parquet(tr_parquet)
    ted = pd.read_parquet(te_parquet)
    X = feats()
    tr_ids = [str(g) for g in trd["genome_id"]]
    te_ids = [str(g) for g in ted["genome_id"]]
    Xtr, Xte = X.loc[tr_ids].to_numpy(np.float32), X.loc[te_ids].to_numpy(np.float32)
    ytr = trd[TARGET].to_numpy(float)
    yte = ted[TARGET].to_numpy(float)
    yb = np.array([1 if s in ("Resistant", "Nonsusceptible") else
                   0 if s in ("Susceptible", "Susceptible-dependent") else -1
                   for s in ted["pheno"].fillna("")], dtype=int)
    has_pheno = (yb >= 0).sum() >= 30 and len(set(yb[yb >= 0])) == 2
    rows = []
    for mname, (fn, scale) in models().items():
        try:
            yh = fit_predict(fn, scale, Xtr, ytr, Xte)
        except Exception:  # noqa: BLE001
            continue
        sd_y = float(np.std(yte))
        sd_h = float(np.std(yh))
        # (a)(b)
        r2_raw = r2_score(yte, yh)
        yh_mean = yh + (yte.mean() - yh.mean())
        r2_mean = r2_score(yte, yh_mean)
        # (c) 最优仿射（oracle，在测试集上拟合 b）
        if sd_h > 1e-9 and sd_y > 1e-9:
            b = float(np.cov(yh, yte, ddof=0)[0, 1] / np.var(yh))
            a = float(yte.mean() - b * yh.mean())
            r2_aff = r2_score(yte, a + b * yh)
            pear = float(pearsonr(yte, yh)[0])
        else:
            r2_aff, pear = np.nan, np.nan
        # (d) k-shot 本地重标定（目标物种内 k 个已标注，其余评估）
        shot = {}
        rng = np.random.RandomState(SEED)
        for k in K_LIST:
            vals = []
            for rep in range(5):
                if k >= len(yte) - 20:
                    break
                idx = rng.permutation(len(yte))
                cal, test = idx[:k], idx[k:]
                if np.std(yh[cal]) < 1e-9:
                    continue
                bb = float(np.cov(yh[cal], yte[cal], ddof=0)[0, 1] / np.var(yh[cal]))
                aa = float(yte[cal].mean() - bb * yh[cal].mean())
                vals.append(r2_score(yte[test], aa + bb * yh[test]))
            shot[f"r2_shot{k}"] = float(np.mean(vals)) if vals else np.nan
        auc = np.nan
        if has_pheno:
            try:
                auc = roc_auc_score(yb[yb >= 0], yh[yb >= 0])
            except Exception:  # noqa: BLE001
                auc = np.nan
        rows.append(dict(drug=drug, species=held, model=mname, n_train=len(trd),
                         n_test=len(ted), n_train_species=trd["species"].nunique(),
                         r2_raw=r2_raw, r2_mean=r2_mean, r2_affine_oracle=r2_aff,
                         pearson=pear,
                         spearman=(spearmanr(yte, yh).correlation
                                   if sd_h > 1e-9 else np.nan),
                         kendall=(kendalltau(yte, yh).correlation
                                  if sd_h > 1e-9 else np.nan),
                         conc=concordance(yte, yh),
                         auc=auc, pred_sd=sd_h, true_sd=sd_y,
                         sd_ratio=(sd_h / sd_y if sd_y > 1e-9 else np.nan),
                         **shot))
    return rows


def main():
    lab = pd.read_parquet(DATA / "labels_mic.parquet")
    lab = lab[lab["genotype_hit"] == 1].dropna(subset=[TARGET]).copy()
    shared = (lab.groupby(["antibiotic", "species"])["genome_id"].nunique()
              .reset_index().rename(columns={"genome_id": "g"}))
    ok = [d for d, grp in shared.groupby("antibiotic") if (grp["g"] >= MIN_UNIT).sum() >= 2]
    l4 = []
    for drug in ok:
        sds = shared[(shared["antibiotic"] == drug) & (shared["g"] >= MIN_UNIT)]["species"].tolist()
        for held in sds:
            trd = lab[(lab["antibiotic"] == drug) & (lab["species"] != held)] \
                .sort_values("val").drop_duplicates("genome_id", keep="first")
            ted = lab[(lab["antibiotic"] == drug) & (lab["species"] == held)] \
                .sort_values("val").drop_duplicates("genome_id", keep="first")
            if len(trd) < 100 or len(ted) < 50:
                continue
            key = f"{drug.replace('/', '_').replace(' ', '')}__{held.replace(' ', '_')}"
            tj, ej = SUBS / f"_tr_{key}.parquet", SUBS / f"_te_{key}.parquet"
            if not tj.exists():
                trd.to_parquet(tj, index=False)
            if not ej.exists():
                ted.to_parquet(ej, index=False)
            l4.append((drug, held, str(tj), str(ej), key))
    print(f"[1] L4 任务 {len(l4)}", flush=True)
    rows = Parallel(n_jobs=N_JOBS, verbose=2)(delayed(analyse)(*t) for t in l4)
    d = pd.DataFrame([r for sub in rows for r in sub])
    d.to_csv(OUT / "correction_ladder.tsv", sep="\t", index=False, encoding="utf-8-sig")

    cols = ["r2_raw", "r2_mean", "r2_affine_oracle", "r2_shot10", "r2_shot25",
            "r2_shot50", "r2_shot100", "spearman", "conc", "auc"]
    print("\n=== 校正阶梯（全模型均值）===")
    print(d[cols].mean().round(3).to_string())
    print("\n=== 按模型 ===")
    print(d.groupby("model")[cols].mean().round(3).to_string())
    print("\n=== 转正比例（阈值 0）===")
    for c in ("r2_raw", "r2_mean", "r2_affine_oracle", "r2_shot25", "r2_shot50", "r2_shot100"):
        print(f"  {c:20s} {(d[c] > 0).mean():.3f}")
    print("\n=== 训练物种数分层 ===")
    g = d.groupby("n_train_species").agg(
        n=("r2_mean", "size"), r2_raw=("r2_raw", "mean"), r2_mean=("r2_mean", "mean"),
        r2_oracle=("r2_affine_oracle", "mean"), auc=("auc", "mean"),
        rho=("spearman", "mean")).reset_index()
    g.to_csv(OUT / "correction_ladder_by_train_species.tsv", sep="\t", index=False,
             encoding="utf-8-sig")
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
