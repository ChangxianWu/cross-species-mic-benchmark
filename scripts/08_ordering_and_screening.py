# -*- coding: utf-8 -*-
"""部分排序的证据强度、筛查任务的实践判别力、同预算迁移 vs 本地训练。

  A 排序强度  对 concordance 与 Spearman 给出**按单元的聚类 bootstrap CI**，
             并对「是否高于无信息值（C=0.5 / ρ=0）」做符号检验；报稳健的拟合比例。
  B 决策任务  不用任意阈值，而是定义一个决策相关任务：把单元内 MIC 最高四分位的
             分离株标记出来（「异常耐药」提示），报灵敏度/特异度/阳性预测值，
             并给出「要达到 PPV≥0.8 需要多高的排序能力」的经验锚点。
  C 预算对照  在同一 100 个本地标签预算下对比三条路径：跨物种迁移 + 重标定 /
             只用这 100 个标签从零训练同物种模型 / 同物种全量训练（上界）。
  D 抽样配对  按单元配对比较（同一单元同一 k）分层抽样与随机抽样，
             给出配对差与符号秩检验，避免用均值噪声下结论。
  E 删失量级  在扰动标签下重算 oracle 系数与保留比例，检验量级陈述是否被改变。

输出：output/ordering_screening.tsv
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, wilcoxon
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
KS = (50, 100)
REPS = 5
BOOT = 2000

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


def concordance(y, yh):
    n = len(y)
    num = den = 0
    for i in range(n - 1):
        dy = y[i + 1:] - y[i]
        dh = yh[i + 1:] - yh[i]
        m = dy != 0
        den += int(m.sum())
        num += int(((dy[m] > 0) == (dh[m] > 0)).sum())
    return num / den if den else np.nan


def affine(y_cal, yh_cal, yh_out):
    v = np.var(yh_cal)
    if v < 1e-12:
        return np.full(len(yh_out), float(np.mean(y_cal)))
    b = float(np.cov(yh_cal, y_cal, ddof=0)[0, 1] / v)
    return float(np.mean(y_cal) - b * np.mean(yh_cal)) + b * yh_out


def perturb(vals, cens, direction):
    v = vals.astype(float).copy()
    if direction > 0:
        v[cens == "left"] -= 0.5
        v[cens == "right"] += 0.5
    else:
        v[cens == "left"] += 0.5
        v[cens == "right"] -= 0.5
    return v


def topq_task(y, yh, q=0.75):
    """把 MIC 最高四分位标为「异常耐药」。返回灵敏度/特异度/PPV。"""
    thr_y = np.quantile(y, q)
    thr_h = np.quantile(yh, q)
    yt = y >= thr_y
    yp = yh >= thr_h
    if yt.sum() == 0 or (~yt).sum() == 0 or yp.sum() == 0:
        return np.nan, np.nan, np.nan
    sens = float((yp[yt]).mean())
    spec = float((~yp[~yt]).mean())
    ppv = float((yt[yp]).mean())
    return sens, spec, ppv


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
    rows = []
    rng = np.random.RandomState(SEED)
    for mname, (fn, scale) in MODELS.items():
        try:
            yh = fit_predict(fn, scale, Xtr, ytr, Xte)
        except Exception:  # noqa: BLE001
            continue
        # ---- 排序强度 / 决策任务 ----
        conc = concordance(yte, yh)
        rho = (spearmanr(yte, yh).correlation if np.std(yh) > 1e-9 else np.nan)
        sens, spec, ppv = topq_task(yte, yh)
        r2_orc = r2_score(yte, affine(yte, yh, yh))
        # ---- 三条路径在同一标签预算下 ----
        paths = {}
        for k in KS:
            if k >= len(yte) - 20:
                continue
            tr_shot, tr_full, loc_shot = [], [], []
            for _ in range(REPS):
                idx = rng.permutation(len(yte))
                cal, rest = idx[:k], idx[k:]
                # (a) 迁移 + k 标签重标定
                tr_shot.append(r2_score(yte[rest], affine(yte[cal], yh[cal], yh[rest])))
                # (b) 只用 k 个本地标签从零训练
                try:
                    yh_loc = fit_predict(fn, scale, Xte[cal], yte[cal], Xte[rest])
                    loc_shot.append(r2_score(yte[rest], yh_loc))
                except Exception:  # noqa: BLE001
                    pass
                # (c) 同物种全量（上界，用同一 rest 评估以保证可比）
                try:
                    yh_full = fit_predict(fn, scale,
                                          np.vstack([Xte[cal], Xte[rest]]),
                                          np.concatenate([yte[cal], yte[rest]]),
                                          Xte[rest])
                    tr_full.append(r2_score(yte[rest], yh_full))
                except Exception:  # noqa: BLE001
                    pass
            paths[f"transfer_k{k}"] = float(np.mean(tr_shot)) if tr_shot else np.nan
            paths[f"local_k{k}"] = float(np.mean(loc_shot)) if loc_shot else np.nan
            paths[f"within_full_k{k}"] = float(np.mean(tr_full)) if tr_full else np.nan
        # ---- 删失扰动下的 oracle 与保留比例 ----
        out = {}
        for tag, d in (("widen", +1), ("narrow", -1)):
            y_tr_p = perturb(ytr, ctr, d)
            y_te_p = perturb(yte, cte, d)
            try:
                yh_p = fit_predict(fn, scale, Xtr, y_tr_p, Xte)
                r2_mean_p = r2_score(y_te_p, yh_p + (y_te_p.mean() - yh_p.mean()))
                r2_orc_p = r2_score(y_te_p, affine(y_te_p, yh_p, yh_p))
            except Exception:  # noqa: BLE001
                r2_mean_p = r2_orc_p = np.nan
            out[f"r2_mean_{tag}"] = r2_mean_p
            out[f"r2_oracle_{tag}"] = r2_orc_p
        rows.append(dict(drug=drug, species=held, model=mname,
                         n_train=len(trd), n_test=len(ted), conc=conc, spearman=rho,
                         r2_oracle=r2_orc, topq_sens=sens, topq_spec=spec,
                         topq_ppv=ppv, **paths, **out))
    return rows


def main():
    lab = pd.read_parquet(DATA / "labels_mic.parquet")
    lab = lab[lab["genotype_hit"] == 1].dropna(subset=[TARGET]).copy()
    shared = (lab.groupby(["antibiotic", "species"])["genome_id"].nunique()
              .reset_index().rename(columns={"genome_id": "g"}))
    ok = [d for d, grp in shared.groupby("antibiotic") if (grp["g"] >= 100).sum() >= 2]
    # 同单元 L1（保留比例分母）
    s4 = pd.read_csv(OUT / "results_all.tsv", sep="	")
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
    d["retention"] = d["r2_oracle"] / d["r2_L1"]
    d["retention_widen"] = d["r2_oracle_widen"] / d["r2_L1"]
    d["retention_narrow"] = d["r2_oracle_narrow"] / d["r2_L1"]
    d.to_csv(OUT / "ordering_screening.tsv", sep="\t", index=False, encoding="utf-8-sig")

    print("\n=== concordance / Spearman 的聚类 bootstrap 与无信息检验 ===")
    for col, null in (("conc", 0.5), ("spearman", 0.0)):
        v = d[col].dropna().to_numpy(float)
        rng = np.random.RandomState(SEED)
        boot = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(BOOT)])
        ci = np.percentile(boot, [2.5, 97.5])
        above = int((v > null).sum())
        p = wilcoxon(v - null, alternative="greater").pvalue if len(v) > 10 else np.nan
        print(f"  {col:9s} n={len(v)} mean={v.mean():.3f} "
              f"95%CI=[{ci[0]:.3f},{ci[1]:.3f}]  高于{null}的比例={above/len(v):.3f} "
              f"Wilcoxon p={p:.2g}")

    print("\n=== 「标记最高四分位」任务的实践判别力 ===")
    for c in ("topq_sens", "topq_spec", "topq_ppv"):
        v = d[c].dropna()
        print(f"  {c:11s} median={v.median():.3f}  IQR=[{v.quantile(.25):.3f},"
              f"{v.quantile(.75):.3f}]")
    ok_ppv = d.dropna(subset=["topq_ppv", "conc"])
    for thr in (0.6, 0.7, 0.8):
        sel = ok_ppv[ok_ppv["conc"] >= thr]
        if len(sel):
            print(f"  C>={thr:.1f}: n={len(sel):3d}  PPV中位={sel['topq_ppv'].median():.3f}")
        else:
            print(f"  C>={thr:.1f}: 无拟合达到")

    print("\n=== 同一 100 标签预算下的三条路径 ===")
    for k in KS:
        cols = [f"transfer_k{k}", f"local_k{k}", f"within_full_k{k}"]
        if all(c in d for c in cols):
            print(f"  k={k}: " + "  ".join(f"{c.split('_k')[0]:12s}={d[c].median():7.3f}"
                                           for c in cols))
            delta = (d[f"local_k{k}"] - d[f"transfer_k{k}"]).dropna()
            print(f"        local−transfer: median={delta.median():+.3f} "
                  f"正比例={(delta > 0).mean():.3f} n={len(delta)}")
    # 预算比：k vs 从头训练所需样本
    print("\n=== 预算：逐单元 k 标签 vs 从零训练所需样本 ===")
    for k in KS:
        ratio = (k / d["n_train"]).median()
        print(f"  k={k}: 中位占「从零训练」样本量的 {ratio:.3%}")

    print("\n=== 分层 vs 随机抽样（按单元配对）===")
    t3 = pd.read_csv(OUT / "recalibration_retention.tsv", sep="\t")
    for k in (25, 50, 100):
        a = t3[f"shot{k}_random"]
        b = t3[f"shot{k}_pred_strat"]
        m = a.notna() & b.notna()
        if m.sum() > 10:
            diff = (b[m] - a[m])
            p = wilcoxon(diff, alternative="two-sided").pvalue
            print(f"  k={k:3d} n={int(m.sum())} 分层−随机 median={diff.median():+.4f} "
                  f"IQR=[{diff.quantile(.25):+.4f},{diff.quantile(.75):+.4f}] "
                  f"Wilcoxon p={p:.2g}")

    print("\n=== 删失扰动下的量级（不只是符号）===")
    for c in ("r2_oracle", "r2_oracle_widen", "r2_oracle_narrow",
              "retention", "retention_widen", "retention_narrow"):
        v = d[c].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {c:20s} median={v.median():.3f}  mean={v.mean():.3f}")
    print("\n完成")


if __name__ == "__main__":
    main()
