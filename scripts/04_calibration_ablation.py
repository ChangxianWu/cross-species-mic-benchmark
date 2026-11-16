# -*- coding: utf-8 -*-
"""校准 / conformal / 特征消融 / 采样密度分解。

四个敏感性分析块：
  A. 校准：把预测值与观测值分箱，拟合校准斜率/截距（Cox 式 logistic 校准
     的回归版），报告 E50/E90 与分箱偏差。基线 = 跨级最差者。
  B. Conformal：split-conformal 区间（绝对残差分位数），报告 L1–L4 各水平的
     覆盖率与平均区间宽度——外推越远、区间越宽才诚实，故同时报告
     「覆盖率 / 宽度」比值。
  C. 特征消融：raw vs mech vs 「仅已知决定子子集」三档，量化特征抽象层对
     跨物种迁移的贡献。
  D. 采样密度分解：把 L1→L4 的 R² 衰减分解为「训练集规模效应」与
     「分布位移效应」——用规模匹配的 L1 子采样做对照（关键对照，
     否则衰减会被误归因于群体结构）。

输出：output/calibration.tsv / conformal.tsv / ablation.tsv /
      density_decomposition.tsv / results_all.tsv
"""
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = Path(BASE) / "data"
OUT = Path(BASE) / "output"
PARTS = Path(BASE) / "cache" / "calibration_parts"   # 断点单元写 cache/，中断后可续跑
PARTS.mkdir(parents=True, exist_ok=True)

SEED = 20261011
MIN_UNIT = 100
N_FOLDS = 5
TARGET = "log2mic"
N_JOBS = 14
ALPHA = 0.10          # conformal 目标误覆盖 10% -> 90% 区间
FEAT_RAW = DATA / "feature_raw.parquet"
FEAT_MECH = DATA / "feature_mech.parquet"
FEAT_KNOWN = DATA / "feature_known.parquet"   # 仅已知决定子（机制核心子集）

# 「已知决定子」子集：只保留教科书级、与药物类直接对应的基因/变异
KNOWN_PAT = (r"blaKPC|blaNDM|blaIMP|blaVIM|blaOXA-4[89]|blaCTX-M|blaSHV|blaTEM|"
             r"mecA|mecC|pbp1a|pbp2b|pbp2x|"
             r"gyrA_|parC_|parE_|"
             r"van[A-Z]|erm\(|mef\(|msr\(|tet\(|tet[A-Z]\b|"
             r"aac\(|aad[A-Z]|aph\(|ant\(|strA|strB|"
             r"sul[0-9]|dfr[A-Z]|cat[A-Z]?|floR|fexA|"
             r"mcr-[0-9]|pmrA|pmrB|armA|rmt[A-Z]|"
             r"cfr|optrA|poxtA|rpoB_|fosA|fosB|fosC|uhpT_")

MODELS = {
    "ridge": (lambda: Ridge(alpha=10.0), True),
    "rf": (lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                         n_jobs=2, random_state=SEED), False),
    "xgb": (None, False),
    "mlp": (lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=150,
                                 early_stopping=True, n_iter_no_change=10,
                                 batch_size=256, learning_rate_init=2e-3,
                                 random_state=SEED), True),
}


def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08,
                        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                        n_jobs=2, random_state=SEED, verbosity=0)


MODELS["xgb"] = (_xgb, False)

_CACHE = {}


def get_feats():
    if not _CACHE:
        for nm, path in (("raw", FEAT_RAW), ("mech", FEAT_MECH), ("known", FEAT_KNOWN)):
            if path.exists():
                df = pd.read_parquet(path)
                df["genome_id"] = df["genome_id"].astype(str)
                _CACHE[nm] = df.set_index("genome_id")
    return _CACHE


def metrics(y, yhat):
    """R² / RMSE / MAE + 偏置校正 R²。

    r2        —— 标准 R²（对测试集均值），跨物种时同时承载「标签均值位移」
                 与「预测失效」两部分；
    r2_shift  —— 把预测整体平移对齐目标均值后的 R²，**只反映排序/离散度
                 是否迁移**，是分离两种失效模式的关键指标；
    rmse_shift—— 平移后的 RMSE。
    """
    if len(y) < 3 or np.std(y) < 1e-9:
        return dict(r2=np.nan, rmse=np.nan, mae=np.nan, r2_shift=np.nan, rmse_shift=np.nan)
    yh_shift = yhat + (float(np.mean(y)) - float(np.mean(yhat)))
    return dict(r2=float(r2_score(y, yhat)),
                rmse=float(np.sqrt(mean_squared_error(y, yhat))),
                mae=float(mean_absolute_error(y, yhat)),
                r2_shift=float(r2_score(y, yh_shift)),
                rmse_shift=float(np.sqrt(mean_squared_error(y, yh_shift))))


def calib_stats(y, yhat, bins=10):
    """校准斜率/截距（y ~ a + b*yhat 的 OLS）+ 分箱偏差 + E50/E90。

    斜率 b > 1 表示预测过于集中（欠离散）；b < 1 表示预测过度发散。
    """
    if len(y) < 20 or np.std(yhat) < 1e-9:
        return dict(cal_slope=np.nan, cal_intercept=np.nan, cal_e50=np.nan,
                    cal_e90=np.nan, cal_bin_mae=np.nan)
    b, a = np.polyfit(yhat, y, 1)
    resid = np.abs(y - yhat)
    # 分箱：按预测值分位分箱，箱内 |均值偏差|
    q = pd.qcut(pd.Series(yhat), q=min(bins, max(2, len(y) // 30)),
                duplicates="drop")
    dev = pd.DataFrame({"q": q, "y": y, "yh": yhat}).groupby("q", observed=True).apply(
        lambda d: abs(d["y"].mean() - d["yh"].mean()))
    return dict(cal_slope=float(b), cal_intercept=float(a),
                cal_e50=float(np.percentile(resid, 50)),
                cal_e90=float(np.percentile(resid, 90)),
                cal_bin_mae=float(np.nanmean(dev.to_numpy())))


def make_folds(group_values, n_folds, seed):
    g = pd.Series(group_values).astype(str)
    sizes = g.value_counts()
    if sizes.size < n_folds * 2 or sizes.max() > 0.8 * len(g):
        return [te for _, te in KFold(n_splits=n_folds, shuffle=True,
                                      random_state=seed).split(np.arange(len(g)))]
    load = np.zeros(n_folds)
    assign = {}
    for grp in sizes.sort_values(ascending=False).index:
        f = int(np.argmin(load))
        assign[grp] = f
        load[f] += sizes[grp]
    fa = g.map(assign).to_numpy()
    return [np.where(fa == k)[0] for k in range(n_folds)]


def fit_predict(mname, Xtr, ytr, Xte):
    fn, scale = MODELS[mname]
    if scale:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = fn()
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def run_unit(args):
    """单元内：L1/L2/L3 × 特征层消融 × 校准 + conformal。"""
    sp, drug, sub_json, key = args
    part = PARTS / f"{key}.json"   # 内容为 JSON 数组（断点单元），勿写成 .parquet
    if part.exists():
        return key, "skip"
    sub = pd.read_parquet(sub_json)
    sub["mlst"] = sub["mlst"].fillna("NA").replace("", "NA")
    sub["country"] = sub["country"].fillna("NA").replace("", "NA")
    gids = sub["genome_id"].tolist()
    y = sub[TARGET].to_numpy(float)
    feats = get_feats()
    rows = []
    for fname, df in feats.items():
        gids_s = [str(g) for g in gids]
        miss = [g for g in gids_s if g not in df.index]
        if miss:
            raise RuntimeError(f"[FATAL] {key}: {len(miss)} 个 genome_id 不在特征矩阵 {fname} 中"
                               f"（前 3 例 {miss[:3]}）")
        X = df.loc[gids_s].to_numpy(np.float32)
        if X.shape[1] == 0:
            continue
        for level, gcol in (("L1", None), ("L2", "mlst"), ("L3", "country")):
            folds = (make_folds(np.arange(len(gids)), N_FOLDS, SEED) if gcol is None
                     else make_folds(sub[gcol].tolist(), N_FOLDS, SEED))
            for fi, te in enumerate(folds):
                tr = np.setdiff1d(np.arange(len(gids)), te)
                if len(tr) < 30 or len(te) < 10:
                    continue
                for mname in MODELS:
                    try:
                        yh = fit_predict(mname, X[tr], y[tr], X[te])
                    except Exception as exc:  # noqa: BLE001
                        rows.append(dict(feature=fname, model=mname, level=level,
                                         species=sp, antibiotic=drug, fold=fi,
                                         err=type(exc).__name__,
                                         n_train=len(tr), n_test=len(te)))
                        continue
                    mt = metrics(y[te], yh)
                    cs = calib_stats(y[te], yh)
                    # split-conformal：用训练集内的残差（新划分 50/50）
                    conf = conformal_eval(X[tr], y[tr], X[te], y[te], mname)
                    rows.append(dict(feature=fname, model=mname, level=level,
                                     species=sp, antibiotic=drug, fold=fi,
                                     n_train=len(tr), n_test=len(te), err="",
                                     **mt, **cs, **conf))
    part.write_text(json.dumps(rows), encoding="utf-8")
    return key, f"{len(rows)}"


def conformal_eval(Xtr, ytr, Xte, yte, mname):
    """split-conformal（半分割训练集做校准集），返回覆盖率与区间宽度。"""
    n = len(ytr)
    if n < 40 or len(yte) < 10:
        return dict(cov=np.nan, width=np.nan, cov_eff=np.nan)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    ncal = n // 2
    cal_i, fit_i = idx[:ncal], idx[ncal:]
    try:
        pred_cal = fit_predict(mname, Xtr[fit_i], ytr[fit_i], Xtr[cal_i])
        pred_te = fit_predict(mname, Xtr[fit_i], ytr[fit_i], Xte)
    except Exception:  # noqa: BLE001
        return dict(cov=np.nan, width=np.nan, cov_eff=np.nan)
    resid = np.abs(ytr[cal_i] - pred_cal)
    k = int(np.ceil((1 - ALPHA) * (len(resid) + 1)))
    k = min(k, len(resid))
    q = np.sort(resid)[k - 1] if len(resid) else np.nan
    cov = float(np.mean(np.abs(yte - pred_te) <= q))
    width = float(2 * q)
    # 有效覆盖率 = 覆盖率 / (区间宽度 / 标签 sd)：宽度越大越容易覆盖
    sd = float(np.std(yte)) if np.std(yte) > 1e-9 else np.nan
    eff = float(cov / (width / sd)) if sd and width and width > 0 else np.nan
    return dict(cov=cov, width=width, cov_eff=eff)


def run_l4(args):
    """L4 跨物种：校准 + conformal + 训练物种数记录。"""
    drug, held, tr_json, te_json, key = args
    part = PARTS / f"L4_{key}.json"
    if part.exists():
        return key, "skip"
    trd = pd.read_parquet(tr_json)
    ted = pd.read_parquet(te_json)
    feats = get_feats()
    ytr = trd[TARGET].to_numpy(float)
    yte = ted[TARGET].to_numpy(float)
    n_train_species = trd["species"].nunique()
    rows = []
    for fname, df in feats.items():
        tr_ids = [str(g) for g in trd["genome_id"]]
        te_ids = [str(g) for g in ted["genome_id"]]
        miss = [g for g in tr_ids + te_ids if g not in df.index]
        if miss:
            raise RuntimeError(f"[FATAL] L4 {key}: {len(miss)} 个 genome_id 缺失"
                               f"（前 3 例 {miss[:3]}）")
        Xtr = df.loc[tr_ids].to_numpy(np.float32)
        Xte = df.loc[te_ids].to_numpy(np.float32)
        if Xtr.shape[1] == 0:
            continue
        for mname in MODELS:
            try:
                yh = fit_predict(mname, Xtr, ytr, Xte)
            except Exception as exc:  # noqa: BLE001
                rows.append(dict(feature=fname, model=mname, level="L4",
                                 species=held, antibiotic=drug, err=type(exc).__name__,
                                 n_train=len(trd), n_test=len(ted),
                                 n_train_species=n_train_species))
                continue
            mt, cs = metrics(yte, yh), calib_stats(yte, yh)
            conf = conformal_eval(Xtr, ytr, Xte, yte, mname)
            rows.append(dict(feature=fname, model=mname, level="L4", species=held,
                             antibiotic=drug, fold=-1, n_train=len(trd),
                             n_test=len(ted), err="", n_train_species=n_train_species,
                             **mt, **cs, **conf))
    part.write_text(json.dumps(rows), encoding="utf-8")
    return key, f"{len(rows)}"


def run_density(args):
    """采样密度分解：规模匹配的 L1 对照。

    对每个 (species, drug)，若 L4 的 held-out 单元有 n_te 个目标样本，
    而训练侧有 n_tr 个（跨物种汇总）；则在同物种 L1 设置下也做一次
    「只用 n_tr 个训练样本」的子采样回归，得到规模匹配的同分布基线。
    ΔR²(L1→L4) 减去 ΔR²(L1_full→L1_matched) 即「分布位移」的净贡献。
    """
    sp, drug, sub_json, key, n_match = args
    part = PARTS / f"DENS_{key}.json"
    if part.exists():
        return key, "skip"
    sub = pd.read_parquet(sub_json).sort_values("val") \
        .drop_duplicates("genome_id", keep="first")
    gids = sub["genome_id"].tolist()
    y = sub[TARGET].to_numpy(float)
    feats = get_feats()
    rng = np.random.RandomState(SEED)
    rows = []
    n = len(gids)
    reps = 5
    for fname, df in feats.items():
        gids_s = [str(g) for g in gids]
        miss = [g for g in gids_s if g not in df.index]
        if miss:
            raise RuntimeError(f"[FATAL] {key}: {len(miss)} 个 genome_id 不在特征矩阵 {fname} 中"
                               f"（前 3 例 {miss[:3]}）")
        X = df.loc[gids_s].to_numpy(np.float32)
        if X.shape[1] == 0:
            continue
        for mname in ("ridge", "rf", "xgb"):
            sc_list = []
            for r in range(reps):
                idx = rng.permutation(n)
                ntr = min(int(n_match), int(n * 0.8))
                if ntr < 30:
                    continue
                tr, te = idx[:ntr], idx[ntr:]
                try:
                    yh = fit_predict(mname, X[tr], y[tr], X[te])
                    mt = metrics(y[te], yh)
                except Exception:  # noqa: BLE001
                    continue
                sc_list.append(mt["r2"])
            if sc_list:
                rows.append(dict(feature=fname, model=mname, level="L1_matched",
                                 species=sp, antibiotic=drug, n_train_matched=n_match,
                                 reps=len(sc_list), r2_mean=float(np.mean(sc_list)),
                                 r2_sd=float(np.std(sc_list))))
    part.write_text(json.dumps(rows), encoding="utf-8")
    return key, f"{len(rows)}"


def build_known_features():
    """仅已知决定子特征层。"""
    gmat = pd.read_parquet(DATA / "genotypes.parquet")
    genes = [c for c in gmat.columns if c != "genome_id"]
    keep = [g for g in genes if re.search(KNOWN_PAT, g, re.I)]
    out = gmat[["genome_id"] + keep].copy()
    out[keep] = out[keep].astype(np.int8)
    out.to_parquet(FEAT_KNOWN, index=False)
    return out.shape, keep[:20]


def main():
    t0 = time.time()
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only in ("all", "features"):
        shp, sample = build_known_features()
        print(f"[1] known 特征层 {shp}（如 {sample[:6]}）", flush=True)
    if only == "features":
        return

    labels = pd.read_parquet(DATA / "labels_mic.parquet")
    lab = labels[labels["genotype_hit"] == 1].dropna(subset=[TARGET]).copy()
    lab["mlst"] = lab["mlst"].fillna("NA").replace("", "NA")
    lab["country"] = lab["country"].fillna("NA").replace("", "NA")

    subdir = DATA / "unit_subsets"
    subdir.mkdir(exist_ok=True)
    grid = lab.groupby(["species", "antibiotic"])["genome_id"].nunique().reset_index()
    grid = grid[grid["genome_id"] >= MIN_UNIT]

    # ---- 单元级：校准 / conformal / 消融 ----
    tasks = []
    for u in grid.itertuples(index=False):
        s = lab[(lab["species"] == u.species) & (lab["antibiotic"] == u.antibiotic)] \
            .sort_values("val").drop_duplicates("genome_id", keep="first")
        key = f"{u.species.replace(' ', '_')}__{u.antibiotic.replace('/', '_').replace(' ', '')}"
        js = subdir / f"{key}.parquet"
        if not js.exists():
            s.to_parquet(js, index=False)
        tasks.append((u.species, u.antibiotic, str(js), key))
    print(f"[2] 单元级任务：{len(tasks)}", flush=True)
    Parallel(n_jobs=N_JOBS, verbose=5, batch_size=1)(delayed(run_unit)(t) for t in tasks)

    # ---- L4 ----
    shared = lab.groupby(["antibiotic", "species"])["genome_id"].nunique() \
        .reset_index().rename(columns={"genome_id": "g"})
    ok_drugs = [d for d, grp in shared.groupby("antibiotic")
                if (grp["g"] >= MIN_UNIT).sum() >= 2]
    l4 = []
    for drug in ok_drugs:
        sds = shared[(shared["antibiotic"] == drug) & (shared["g"] >= MIN_UNIT)]["species"].tolist()
        for held in sds:
            trd = lab[(lab["antibiotic"] == drug) & (lab["species"] != held)] \
                .sort_values("val").drop_duplicates("genome_id", keep="first")
            ted = lab[(lab["antibiotic"] == drug) & (lab["species"] == held)] \
                .sort_values("val").drop_duplicates("genome_id", keep="first")
            if len(trd) < 100 or len(ted) < 50:
                continue
            key = f"{drug.replace('/', '_').replace(' ', '')}__{held.replace(' ', '_')}"
            tj, ej = subdir / f"_tr_{key}.parquet", subdir / f"_te_{key}.parquet"
            trd.to_parquet(tj, index=False)
            ted.to_parquet(ej, index=False)
            l4.append((drug, held, str(tj), str(ej), key))
    print(f"[3] L4 任务：{len(l4)}", flush=True)
    Parallel(n_jobs=N_JOBS, verbose=5, batch_size=1)(delayed(run_l4)(t) for t in l4)

    # ---- 采样密度分解 ----
    print("[4] 采样密度分解（规模匹配 L1 对照）...", flush=True)
    dens = []
    for drug in ok_drugs:
        sds = shared[(shared["antibiotic"] == drug) & (shared["g"] >= MIN_UNIT)]["species"].tolist()
        for held in sds:
            trd = lab[(lab["antibiotic"] == drug) & (lab["species"] != held)] \
                .sort_values("val").drop_duplicates("genome_id", keep="first")
            if len(trd) < 100:
                continue
            s = lab[(lab["species"] == held) & (lab["antibiotic"] == drug)] \
                .sort_values("val").drop_duplicates("genome_id", keep="first")
            if len(s) < MIN_UNIT:
                continue
            key = f"{drug.replace('/', '_').replace(' ', '')}__{held.replace(' ', '_')}"
            js = subdir / f"{held.replace(' ', '_')}__{drug.replace('/', '_').replace(' ', '')}.json"
            if not js.exists():
                s.to_parquet(js, index=False)
            dens.append((held, drug, str(js), key, len(trd)))
    Parallel(n_jobs=N_JOBS, verbose=5, batch_size=1)(delayed(run_density)(t) for t in dens)

    # ---- 汇总 ----
    def collect(prefix=""):
        rows = []
        for p in sorted(PARTS.glob("*.json")):
            if prefix and not p.name.startswith(prefix):
                continue
            if not prefix and p.name.startswith("DENS_"):
                continue
            rows.extend(json.loads(p.read_text(encoding="utf-8")))
        return pd.DataFrame(rows)

    allr = collect()
    allr = allr[allr["err"] == ""] if "err" in allr else allr
    allr.to_csv(OUT / "results_all.tsv", sep="\t", index=False, encoding="utf-8-sig")

    cal = (allr.groupby(["feature", "model", "level"])
           .agg(n=("r2", "size"), r2=("r2", "mean"),
                cal_slope=("cal_slope", "mean"), cal_intercept=("cal_intercept", "mean"),
                cal_e50=("cal_e50", "mean"), cal_e90=("cal_e90", "mean"),
                cal_bin_mae=("cal_bin_mae", "mean")).reset_index())
    cal.to_csv(OUT / "calibration.tsv", sep="\t", index=False, encoding="utf-8-sig")

    conf = (allr.groupby(["feature", "model", "level"])
            .agg(n=("cov", "size"), cov=("cov", "mean"), width=("width", "mean"),
                 cov_eff=("cov_eff", "mean"), rmse=("rmse", "mean")).reset_index())
    conf["target_cov"] = 1 - ALPHA
    conf.to_csv(OUT / "conformal.tsv", sep="\t", index=False, encoding="utf-8-sig")

    abl = (allr.groupby(["feature", "model", "level"])
           .agg(r2=("r2", "mean"), rmse=("rmse", "mean"), mae=("mae", "mean")).reset_index())
    abl.to_csv(OUT / "ablation.tsv", sep="\t", index=False, encoding="utf-8-sig")

    d = collect("DENS_")
    if len(d):
        d.to_csv(OUT / "density_decomposition.tsv", sep="\t", index=False,
                 encoding="utf-8-sig")

    print("\n=== 校准（slope 越接近 1 越好；L4 预期偏离最大）===", flush=True)
    print(cal.pivot_table(index=["feature", "model"], columns="level",
                          values="cal_slope").round(3).to_string(), flush=True)
    print("\n=== Conformal 覆盖率（目标 0.90）与区间宽度 ===", flush=True)
    print(conf.pivot_table(index=["feature", "model"], columns="level",
                           values="cov").round(3).to_string(), flush=True)
    print(f"\n总耗时 {round(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    main()
