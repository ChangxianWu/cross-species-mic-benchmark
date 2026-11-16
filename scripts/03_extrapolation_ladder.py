# -*- coding: utf-8 -*-
"""四级外推阶梯基准（并行 + 分段落盘 + 可断点续跑）。

设计：
  分析单元 = (species, antibiotic)，要求 >= MIN_UNIT 个具基因型的基因组。
  目标     = log2(MIC mg/L)。
  特征层   = raw（AMRFinderPlus 原始基因名，物种特异）
             mech（抗性机制类别聚合，跨物种共享词表）
  四级阶梯 = L1 随机 / L2 留 MLST 谱系 / L3 留国家 / L4 留物种
  模型族   = ridge / rf / xgb / mlp（同一组超参，不做逐单元调参，避免
             「调参自由度」污染外推对比）

工程约束：
  - 单元层并行（joblib），单进程内模型级并行交给各模型自身；
  - 长跑任务分段落盘（每单元一个 json），支持断点续跑（已存在即跳过）；
  - 单元内多模型共享同一份折划分，保证模型间可比；
  - 特征矩阵用 parquet 交换 + 每进程惰性缓存（不用 pickle）。

输出：
  cache/ladder_parts/<unit_key>.json   单单元结果（断点单元）
  output/ladder_results.tsv            汇总（长表）
  output/ladder_summary.tsv            按 (feature, model, level) 汇总
  output/unit_inventory.tsv            单元清单
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
from scipy.stats import spearmanr
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
PARTS = Path(BASE) / "cache" / "ladder_parts"   # 断点单元写 cache/，中断后可续跑
PARTS.mkdir(parents=True, exist_ok=True)

SEED = 20261011
MIN_UNIT = 100
N_FOLDS = 5
TARGET = "log2mic"
N_JOBS = 14
FEAT_RAW = DATA / "feature_raw.parquet"
FEAT_MECH = DATA / "feature_mech.parquet"

# ---------- 机制类别（跨物种共享词表） ----------
MECH_RULES = [
    ("carbapenemase", r"blaKPC|blaNDM|blaIMP|blaVIM|blaOXA-4[89]|blaOXA-18[12]|blaOXA-23[2]|blaGES|blaSPM|blaSIM|blaGIM"),
    ("ESBL", r"blaCTX-M|blaSHV-(2|5|12|27|28|30|31)|blaTEM-(3|4|5|6|10|12|20|24|52|87|116)|blaPER|blaVEB|blaBEL"),
    ("narrow_betalactamase", r"blaSHV|blaTEM|blaOXA|blaEC|blaZ|blaI|blaR1|blaC|blaCARB"),
    ("ampC", r"blaCMY|blaDHA|blaFOX|blaMOX|blaACT|blaMIR|blaACC|ampC"),
    ("pbp_alteration", r"pbp1a|pbp2b|pbp2x|pbp2a|pbp3|mecA|mecC|mecR1"),
    ("aminoglycoside_modifying", r"aac\(|aad[A-Z]|aph\(|ant\(|strA|strB|aadA|aadB|aadD"),
    ("aminoglycoside_target", r"armA|rmt[A-Z]|npmA"),
    ("fluoroquinolone_target", r"gyrA_|parC_|parE_|gyrB_"),
    ("quinolone_plasmid", r"qnr[A-Z]|qepA|oqxA|oqxB|aac\(6'\)-Ib-cr"),
    ("tetracycline", r"tet\(|tet[A-Z]\b|otrA|tcr3"),
    ("macrolide_lincosamide", r"erm\(|mef\(|msr\(|mph\(|mrx\(|lnu\(|vga|lsa\("),
    ("phenicol", r"cat[A-Z]?|catB|cmlA|floR|fexA|pexA|cmx"),
    ("sulfonamide_trimethoprim", r"sul[0-9]|dfr[A-Z]"),
    ("vancomycin", r"van[A-Z]-|van[A-Z]\b|vanH|vanR|vanS|vanW|vanX|vanY|vanZ"),
    ("colistin_polymyxin", r"mcr-[0-9]|pmrA|pmrB|phoP|phoQ|arnA|eptA|lpx"),
    ("linezolid", r"cfr|optrA|poxtA|rplC_|rplD_"),
    ("rifamycin", r"rpoB_|arr-|arr\b"),
    ("fosfomycin", r"fosA|fosB|fosC|fosX|murA|uhpT_|glpT"),
    ("efflux_or_porin", r"acr[A-F]|mex[A-Z]|mexa|mepA|emrD|mdt[A-Z]|mdfA|tolC|ompK|ompC|ompF|oprD|amvA"),
    ("other_or_unknown", r".*"),
]
MECH_NAMES = [m[0] for m in MECH_RULES]


def mech_of(gene):
    for name, pat in MECH_RULES:
        if re.search(pat, gene, re.I):
            return name
    return "other_or_unknown"


_CACHE = {}


def get_feats():
    """每进程惰性加载特征矩阵；返回 {name: DataFrame(index=genome_id)}。"""
    if not _CACHE:
        for nm, path in (("raw", FEAT_RAW), ("mech", FEAT_MECH)):
            df = pd.read_parquet(path)
            df["genome_id"] = df["genome_id"].astype(str)
            _CACHE[nm] = df.set_index("genome_id")
    return _CACHE


def make_models():
    def _xgb():
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08,
                                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                n_jobs=2, random_state=SEED, verbosity=0)
        except Exception:  # noqa: BLE001
            return RandomForestRegressor(n_estimators=200, n_jobs=2, random_state=SEED)
    return {
        "ridge": (lambda: Ridge(alpha=10.0), True),
        "rf": (lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                             n_jobs=2, random_state=SEED), False),
        "xgb": (_xgb, False),
        "mlp": (lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=150,
                                     early_stopping=True, n_iter_no_change=10,
                                     batch_size=256, learning_rate_init=2e-3,
                                     random_state=SEED), True),
    }


def metrics(y, yhat):
    if len(y) < 3 or np.std(y) < 1e-9:
        return dict(r2=np.nan, rmse=np.nan, mae=np.nan, spearman=np.nan)
    r2 = float(r2_score(y, yhat))
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    mae = float(mean_absolute_error(y, yhat))
    rho = float("nan")
    if np.std(yhat) > 1e-12:
        c = spearmanr(y, yhat).correlation
        rho = float(c) if c == c else float("nan")
    return dict(r2=r2, rmse=rmse, mae=mae, spearman=rho)


def make_folds(group_values, n_folds, seed):
    """组感知折划分：同一 group 落同一折；组数不足退化为随机折。"""
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


def run_unit(args):
    """单 (species, drug) 单元：L1/L2/L3 × 2 特征层 × 4 模型。"""
    sp, drug, sub_json, unit_key = args
    part = PARTS / f"{unit_key}.json"
    if part.exists():
        return unit_key, "skip"
    sub = pd.read_parquet(sub_json)
    sub["mlst"] = sub["mlst"].fillna("NA").replace("", "NA")
    sub["country"] = sub["country"].fillna("NA").replace("", "NA")
    gids = sub["genome_id"].tolist()
    y = sub[TARGET].to_numpy(float)
    feats = get_feats()
    models = make_models()
    rows = []
    for fname, Xall_df in feats.items():
        gids_s = [str(g) for g in gids]
        miss = [g for g in gids_s if g not in Xall_df.index]
        if miss:
            raise RuntimeError(
                f"[FATAL] {unit_key}: {len(miss)} 个 genome_id 不在特征矩阵中"
                f"（前 3 例 {miss[:3]}）——索引类型不匹配或数据缺失，"
                "静默填零会把模型退化成常数预测。")
        Xall = Xall_df.loc[gids_s].to_numpy(np.float32)
        if Xall.shape[1] == 0:
            continue
        for level, gcol in (("L1", None), ("L2", "mlst"), ("L3", "country")):
            folds = (make_folds(np.arange(len(gids)), N_FOLDS, SEED) if gcol is None
                     else make_folds(sub[gcol].tolist(), N_FOLDS, SEED))
            for fi, te in enumerate(folds):
                tr = np.setdiff1d(np.arange(len(gids)), te)
                if len(tr) < 30 or len(te) < 10:
                    continue
                base = np.full(len(te), y[tr].mean())
                mb = metrics(y[te], base)
                for mname, (fn, scale) in models.items():
                    try:
                        if scale:
                            sc = StandardScaler().fit(Xall[tr])
                            Xtr, Xte = sc.transform(Xall[tr]), sc.transform(Xall[te])
                        else:
                            Xtr, Xte = Xall[tr], Xall[te]
                        m = fn()
                        m.fit(Xtr, y[tr])
                        yh = m.predict(Xte)
                        mt, err = metrics(y[te], yh), ""
                        constant_pred = int(float(np.std(yh)) < 1e-9)
                    except Exception as exc:  # noqa: BLE001
                        mt = dict(r2=np.nan, rmse=np.nan, mae=np.nan, spearman=np.nan)
                        err, constant_pred = type(exc).__name__, np.nan
                    rows.append(dict(feature=fname, model=mname, level=level,
                                     species=sp, antibiotic=drug, fold=fi,
                                     n_train=len(tr), n_test=len(te), err=err,
                                     constant_pred=constant_pred,
                                     pred_sd=float(np.std(yh)) if err == "" else np.nan,
                                     baseline_r2=mb["r2"], baseline_rmse=mb["rmse"],
                                     **mt))
    part.write_text(json.dumps(rows), encoding="utf-8")
    return unit_key, f"{len(rows)}"


def run_l4(args):
    """L4 跨物种：同一药物，留出整个物种。"""
    drug, held, tr_json, te_json, key = args
    part = PARTS / f"L4_{key}.json"
    if part.exists():
        return key, "skip"
    trd = pd.read_parquet(tr_json)
    ted = pd.read_parquet(te_json)
    feats = get_feats()
    models = make_models()
    ytr = trd[TARGET].to_numpy(float)
    yte = ted[TARGET].to_numpy(float)
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
        shared = int(((Xtr.sum(0) > 0) & (Xte.sum(0) > 0)).sum())
        mb = metrics(yte, np.full(len(yte), ytr.mean()))
        for mname, (fn, scale) in models.items():
            try:
                if scale:
                    sc = StandardScaler().fit(Xtr)
                    A, B = sc.transform(Xtr), sc.transform(Xte)
                else:
                    A, B = Xtr, Xte
                m = fn()
                m.fit(A, ytr)
                yh = m.predict(B)
                mt, err = metrics(yte, yh), ""
                constant_pred = int(float(np.std(yh)) < 1e-9)
            except Exception as exc:  # noqa: BLE001
                mt = dict(r2=np.nan, rmse=np.nan, mae=np.nan, spearman=np.nan)
                err, constant_pred = type(exc).__name__, np.nan
            rows.append(dict(feature=fname, model=mname, level="L4", species=held,
                             antibiotic=drug, fold=-1, n_train=len(trd),
                             n_test=len(ted), err=err, shared_cols=shared,
                             n_cols=int(Xtr.shape[1]), constant_pred=constant_pred,
                             pred_sd=float(np.std(yh)) if err == "" else np.nan,
                             train_species="|".join(sorted(trd["species"].unique())),
                             baseline_r2=mb["r2"], baseline_rmse=mb["rmse"], **mt))
    part.write_text(json.dumps(rows), encoding="utf-8")
    return key, f"{len(rows)}"


def build_features():
    """构建两种特征抽象层并落盘为 parquet（raw / mech）。"""
    gmat = pd.read_parquet(DATA / "genotypes.parquet")
    genes = [c for c in gmat.columns if c != "genome_id"]
    Xraw = gmat[["genome_id"] + genes].copy()
    Xraw[genes] = Xraw[genes].astype(np.int8)
    Xraw.to_parquet(FEAT_RAW, index=False)

    mech_map = {g: mech_of(g) for g in genes}
    mech = pd.DataFrame({"genome_id": gmat["genome_id"].values})
    for m in MECH_NAMES:
        cols = [g for g in genes if mech_map[g] == m]
        mech[m] = (gmat[cols].sum(axis=1) > 0).astype(np.int8) if cols else np.int8(0)
    mech["n_resistance_genes"] = gmat[genes].sum(axis=1).astype(np.float32)
    mech.to_parquet(FEAT_MECH, index=False)
    return Xraw.shape, mech.shape


def main():
    t0 = time.time()
    only = sys.argv[1] if len(sys.argv) > 1 else "all"

    if only in ("all", "features"):
        shp = build_features()
        print(f"[1] 特征矩阵 raw={shp[0]} mech={shp[1]}", flush=True)
    if only == "features":
        return

    labels = pd.read_parquet(DATA / "labels_mic.parquet")
    lab = labels[labels["genotype_hit"] == 1].dropna(subset=[TARGET]).copy()
    lab["mlst"] = lab["mlst"].fillna("NA").replace("", "NA")
    lab["country"] = lab["country"].fillna("NA").replace("", "NA")

    inv = (lab.groupby(["species", "antibiotic"])
           .agg(n=("genome_id", "size"), g=("genome_id", "nunique"),
                n_lineage=("mlst", "nunique"), n_country=("country", "nunique"),
                y_sd=(TARGET, "std")).reset_index())
    inv.to_csv(OUT / "unit_inventory.tsv", sep="\t", index=False, encoding="utf-8-sig")
    units = inv[inv["g"] >= MIN_UNIT].copy()
    print(f"[2] 可分析单元: {len(units)}", flush=True)

    subdir = DATA / "unit_subsets"
    subdir.mkdir(exist_ok=True)
    tasks = []
    for u in units.itertuples(index=False):
        s = lab[(lab["species"] == u.species) & (lab["antibiotic"] == u.antibiotic)]
        s = s.sort_values("val").drop_duplicates("genome_id", keep="first")
        if len(s) < MIN_UNIT:
            continue
        key = f"{u.species.replace(' ', '_')}__{u.antibiotic.replace('/', '_').replace(' ', '')}"
        js = subdir / f"{key}.parquet"
        s.to_parquet(js, index=False)
        tasks.append((u.species, u.antibiotic, str(js), key))

    print(f"[3] L1–L3 并行：{len(tasks)} 单元", flush=True)
    Parallel(n_jobs=N_JOBS, verbose=5, batch_size=1)(delayed(run_unit)(t) for t in tasks)
    print(f"[4] L1–L3 完成（{round(time.time()-t0)}s）", flush=True)

    print("[5] L4 跨物种任务构建 ...", flush=True)
    shared = (lab.groupby(["antibiotic", "species"])["genome_id"].nunique()
              .reset_index().rename(columns={"genome_id": "g"}))
    ok_drugs = [d for d, grp in shared.groupby("antibiotic")
                if (grp["g"] >= MIN_UNIT).sum() >= 2]
    l4_tasks = []
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
            l4_tasks.append((drug, held, str(tj), str(ej), key))
    print(f"    可对齐药物 {len(ok_drugs)}；L4 任务 {len(l4_tasks)}", flush=True)
    Parallel(n_jobs=N_JOBS, verbose=5, batch_size=1)(delayed(run_l4)(t) for t in l4_tasks)
    print(f"[6] L4 完成（{round(time.time()-t0)}s）", flush=True)

    rows = []
    for p in sorted(PARTS.glob("*.json")):
        rows.extend(json.loads(p.read_text(encoding="utf-8")))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "ladder_results.tsv", sep="\t", index=False, encoding="utf-8-sig")

    good = res[(res["err"] == "") & res["r2"].notna()]
    summ = (good.groupby(["feature", "model", "level"])
            .agg(n_runs=("r2", "size"), r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
                 r2_med=("r2", "median"), rmse_mean=("rmse", "mean"),
                 mae_mean=("mae", "mean"), rho_mean=("spearman", "mean"),
                 base_r2_mean=("baseline_r2", "mean"))
            .reset_index())
    summ.to_csv(OUT / "ladder_summary.tsv", sep="\t", index=False, encoding="utf-8-sig")
    print("\n=== 阶梯汇总（R² 均值）===", flush=True)
    print(summ.pivot_table(index=["feature", "model"], columns="level",
                           values="r2_mean").round(3).to_string(), flush=True)
    print(f"\n总耗时 {round(time.time()-t0)}s；结果行 {len(res)}", flush=True)


if __name__ == "__main__":
    main()
