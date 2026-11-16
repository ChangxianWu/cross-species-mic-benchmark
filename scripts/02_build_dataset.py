# -*- coding: utf-8 -*-
"""构建「MIC 定量标签 + 基因型特征」配对数据集。

连接策略：
  BV-BRC genome_amr.genome_id
    -> BV-BRC genome.biosample_accession / assembly_accession
      -> NCBI Pathogen Detection *.amr.metadata.tsv 的 biosample_acc / asm_acc
        -> 该行的 AMR_genotypes（AMRFinderPlus 输出）+ computed_types（MLST）

为什么走 biosample 而不是 asm_acc：BV-BRC 有大量历史基因组无 assembly_accession
（Kp 抽样 200 中仅 2 个有），但 biosample 覆盖率接近 100%，且与 PD 表命中率高
（抽样实测 Kp 174/200、Sp 200/200、Pa 195/200、Ec 121/200、Sa 130/200、Ef 45/200）。

PD 表放置：data/pd/<taxon_id>.amr.metadata.tsv（六个物种各一份）。

输出：
  data/labels_mic.parquet      定量 MIC 标签（长表，含截尾标记）
  data/genotypes.parquet       基因型矩阵（genome_id × AMR 基因二值）
  data/genome_meta.parquet     谱系/地理/年份元数据
  output/join_report.tsv       连接率报告
"""
import csv
import json
import os
import re
import sys
import time
import urllib.parse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from net_guard import open_url  # noqa: E402

API = "https://www.bv-brc.org/api/"
HEADERS = {"User-Agent": "cross-species-mic/1.0 (research use)",
           "Accept": "application/json"}
DATE = "2026-10-11"
DATA = Path(BASE) / "data"
OUT = Path(BASE) / "output"
PD = Path(BASE) / "data" / "pd"

SPECIES = [
    ("Klebsiella pneumoniae", 573, "573"),
    ("Escherichia coli", 562, "562"),
    ("Staphylococcus aureus", 1280, "1280"),
    ("Pseudomonas aeruginosa", 287, "287"),
    ("Streptococcus pneumoniae", 1313, "1313"),
    ("Enterococcus faecium", 1352, "1352"),
]
# PD 的 AMR 表统一按物种 taxon id 放置：data/pd/<taxon_id>.amr.metadata.tsv
SPECIES = [(n, t, k, PD / f"{k}.amr.metadata.tsv") for n, t, k in SPECIES]

# 报告用
report = []


def log(*a):
    print(*a, flush=True)


def get(url, tries=4):
    last = None
    for a in range(tries):
        try:
            with open_url(url, headers=HEADERS, timeout=300) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            if a < tries - 1:
                time.sleep(4 * (a + 1))
    raise RuntimeError(f"GET 失败：{url} -> {last}")


def enc(cons):
    return "&".join(urllib.parse.quote(c, safe="(),.") for c in cons)


# ---------------- 1. 标签表清洗 ----------------

def clean_labels(df):
    """把 measurement 解析成 log2 MIC + 截尾方向。

    规则（保守，全部可复核）：
      - 仅保留 measurement_unit == "mg/L"（排除 mm 纸片法、单位缺失行）；
      - measurement_value 必须是正数（转 float）；
      - sign 映射：'' / '=' -> 精确；'<=' / '<' -> 左截尾（真值 ≤ 值）；
                   '>' / '>=' -> 右截尾（真值 ≥ 值）；
      - log2 变换后作为回归目标。
    """
    d = df.copy().rename(columns={"_lab_method": "lab_method"})
    d["unit"] = d["measurement_unit"].fillna("")
    d["sign"] = d["measurement_sign"].fillna("").str.strip()
    d["val"] = pd.to_numeric(d["measurement_value"], errors="coerce")
    n0 = len(d)
    keep = (d["unit"] == "mg/L") & d["val"].notna() & (d["val"] > 0)
    d = d[keep].copy()
    d["cens"] = np.select(
        [d["sign"].isin(["<", "<="]), d["sign"].isin([">", ">="])],
        ["left", "right"], default="exact")
    d["log2mic"] = np.log2(d["val"])
    log(f"    标签清洗：{n0} -> {len(d)} 行（保留 mg/L 且数值为正）")
    return d[["species", "taxon_id", "genome_id", "antibiotic", "val", "sign",
              "cens", "log2mic", "resistant_phenotype", "lab_method", "pmid"]]


# ---------------- 2. PD 表基因型 ----------------

GEN_RE = re.compile(r"[A-Za-z0-9_'()./\-]+")


def parse_pd_amr_table(path, taxon):
    """读 PD AMR 表，返回 {biosample_acc: (asm_acc, mlst, genotype_set)}。"""
    if not path.exists():
        return None
    out = {}
    with path.open("r", encoding="utf-8", newline="", errors="replace") as f:
        rd = csv.reader(f, delimiter="\t")
        head = next(rd)
        ix = {h: i for i, h in enumerate(head)}
        ib, ia = ix["biosample_acc"], ix["asm_acc"]
        ig = ix["AMR_genotypes"]
        ic = ix["computed_types"] if "computed_types" in ix else None
        igeo = ix.get("geo_loc_name")
        icol = ix.get("collection_date")
        n = 0
        for row in rd:
            if len(row) <= max(ib, ia, ig):
                continue
            if taxon is not None and "species_taxid" in ix:
                if len(row) > ix["species_taxid"] and row[ix["species_taxid"]] not in ("", "NULL", str(taxon)):
                    continue
            bs = row[ib]
            if not bs or bs == "NULL":
                continue
            gs = row[ig]
            genes = set()
            if gs and gs != "NULL":
                for tok in GEN_RE.findall(gs):
                    # 去掉 AMRFinderPlus 后缀修饰（PARTIAL_END_OF_CONTIG / HMM / INTERNAL_STOP 等）
                    if tok.upper() in ("PARTIAL_END_OF_CONTIG", "HMM", "INTERNAL_STOP",
                                       "PARTIAL", "POINT", "BLAST", "EXACT", "PARTIAL_START"):
                        continue
                    if len(tok) < 2:
                        continue
                    genes.add(tok)
            mlst = ""
            if ic is not None and len(row) > ic and row[ic] not in ("", "NULL"):
                for part in re.split(r"[;,]", row[ic]):
                    p = part.strip()
                    if p.lower().startswith("mlst"):
                        mlst = p
                        break
            out[bs] = (row[ia], mlst, genes,
                       row[igeo] if igeo is not None and len(row) > igeo else "",
                       row[icol] if icol is not None and len(row) > icol else "")
            n += 1
    return out


def main():
    # ---------- 标签 ----------
    frames = []
    for name, tid, key, _ in SPECIES:
        f = DATA / f"mic_{key}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    raw = pd.concat(frames, ignore_index=True)
    log(f"[1/5] 原始 MIC 行：{len(raw)}")
    labels = clean_labels(raw)
    labels = labels.drop_duplicates(subset=["genome_id", "antibiotic", "lab_method"])
    log(f"    去重后：{len(labels)}")

    # ---------- BV-BRC 基因组元数据 ----------
    log("[2/5] 取 BV-BRC 基因组元数据（biosample/asm/mlst/country/year）...")
    all_ids = sorted(labels["genome_id"].unique().tolist())
    meta_rows = []
    for i in range(0, len(all_ids), 200):
        chunk = all_ids[i:i + 200]
        url = API + "genome/?" + enc([
            "in(genome_id,(" + ",".join(chunk) + "))"]) + \
            "&select(genome_id,assembly_accession,biosample_accession,sra_accession,mlst," \
            "cgmlst_id,isolation_country,collection_year,genome_quality,isolation_source," \
            "host_name,bioproject_accession,genome_length,plasmids,genus)&limit(400)"
        meta_rows.extend(get(url))
        if i % 2000 == 0:
            log(f"    {i}/{len(all_ids)}")
        time.sleep(0.3)
    meta = pd.DataFrame(meta_rows).drop_duplicates(subset=["genome_id"])
    log(f"    元数据：{len(meta)} 基因组")
    meta.to_parquet(DATA / "genome_meta.parquet", index=False)

    ids_set = set(all_ids)
    labels = labels[labels["genome_id"].isin(ids_set)]

    # ---------- 连接 PD 基因型 ----------
    log("[3/5] 连接 NCBI PD AMRFinderPlus 基因型（biosample 主键）...")
    pd_all = {}
    for name, tid, key, path in SPECIES:
        t0 = time.time()
        m = parse_pd_amr_table(path, tid if tid != 562 else None)
        if m is None:
            log(f"    {name}: 本地表缺失 -> 跳过（{path.name}）")
            report.append([name, "本地表缺失", "", "", ""])
            continue
        log(f"    {name}: PD 表解析 {len(m)} 记录（{round(time.time()-t0,1)}s）")
        pd_all[name] = m

    # ---------- 拼装 ----------
    log("[4/5] 拼装配对数据集 ...")
    mm = meta.set_index("genome_id").to_dict("index")
    rows, gen_rows = [], []
    gene_vocab = defaultdict(int)
    for r in labels.itertuples(index=False):
        gi = r.genome_id
        m = mm.get(gi)
        if m is None:
            continue
        bs = m.get("biosample_accession")
        pdt = pd_all.get(r.species)
        hit = pdt.get(bs) if (pdt and bs) else None
        genes = hit[2] if hit else set()
        for g in genes:
            gene_vocab[g] += 1
        rows.append(dict(genome_id=gi, species=r.species, taxon_id=r.taxon_id,
                         antibiotic=r.antibiotic, val=r.val, sign=r.sign, cens=r.cens,
                         log2mic=r.log2mic, pheno=r.resistant_phenotype,
                         lab_method=r.lab_method,
                         biosample=bs or "", asm=(m.get("assembly_accession") or ""),
                         mlst=(m.get("mlst") or ""), cgmlst=(m.get("cgmlst_id") or ""),
                         country=(m.get("isolation_country") or ""),
                         year=m.get("collection_year") or "",
                         iso_source=(m.get("isolation_source") or ""),
                         host=(m.get("host_name") or ""),
                         genotype_hit=int(hit is not None),
                         cluster=(hit[1] if hit else ""),
                         geo=(hit[3] if hit else ""),
                         coll_date=(hit[4] if hit else "")))
        gen_rows.append((gi, genes))

    paired = pd.DataFrame(rows)
    log(f"    配对行：{len(paired)}；有基因型的行：{int(paired['genotype_hit'].sum())}")

    # 基因矩阵（仅保留出现在 >=20 个基因组的基因，控制维度）
    MIN_PREV = 20
    keep_genes = sorted(g for g, c in gene_vocab.items() if c >= MIN_PREV)
    log(f"    基因词表：{len(gene_vocab)} 去重；>= {MIN_PREV} 次保留 {len(keep_genes)}")
    gset = dict(gen_rows)
    gi_list = sorted(set(x[0] for x in gen_rows))
    mat = np.zeros((len(gi_list), len(keep_genes)), dtype=np.int8)
    gidx = {g: j for j, g in enumerate(keep_genes)}
    for i, gi in enumerate(gi_list):
        for g in gset.get(gi, ()):  # noqa: B007
            j = gidx.get(g)
            if j is not None:
                mat[i, j] = 1
    gmat = pd.DataFrame(mat, columns=keep_genes)
    gmat.insert(0, "genome_id", gi_list)
    gmat.to_parquet(DATA / "genotypes.parquet", index=False)
    log(f"    基因型矩阵：{gmat.shape}")

    paired.to_parquet(DATA / "labels_mic.parquet", index=False)
    log("    写入 labels_mic.parquet")

    # ---------- 连接率报告 ----------
    log("[5/5] 连接率报告 ...")
    rep = (paired.groupby("species")
           .agg(rows=("genome_id", "size"),
                genomes=("genome_id", "nunique"),
                drugs=("antibiotic", "nunique"),
                genotype_hit_rate=("genotype_hit", "mean"))
           .reset_index())
    for r in rep.itertuples(index=False):
        log(f"    {r.species:26s} 行={r.rows:>7} 基因组={r.genomes:>6} "
            f"药物={r.drugs:>3} 基因型命中率={r.genotype_hit_rate:.1%}")
        report.append([r.species, "配对统计", str(r.rows), str(r.genomes),
                       f"{r.genotype_hit_rate:.4f}"])
    rp = OUT / "join_report.tsv"
    with rp.open("w", encoding="utf-8-sig", newline="") as f:
        f.write("species\titem\trows\tgenomes/\tgenotype_hit_rate\tdate\n")
        for r in report:
            f.write("\t".join(r) + f"\t{DATE}\n")
    log("写入:", rp)


if __name__ == "__main__":
    main()
