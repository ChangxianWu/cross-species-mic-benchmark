# -*- coding: utf-8 -*-
"""数据抓取：BV-BRC 定量 MIC（六物种全量）。

为什么不用 NCBI Pathogen Detection 的 AST_phenotypes：该字段填充率极低
（E. faecalis 116/15,080 = 0.8%；Campylobacter 4,428/175,089 = 2.5%），
且只有 S/I/R 定性，无法支撑 MIC 定量回归。BV-BRC genome_amr 表
evidence="Laboratory Method" 行给出 measurement_value + measurement_sign +
measurement_unit，是可直接建模的定量标签。

输出：data/mic_<taxon>.parquet（断点续传：已存在即跳过）
清单：data/MANIFEST_mic.tsv
"""
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path

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
DATA.mkdir(parents=True, exist_ok=True)

# 六个目标物种。Acinetobacter 因 MIC=0 记录占比过高被排除。
SPECIES = [
    ("Klebsiella pneumoniae", 573),
    ("Escherichia coli", 562),
    ("Staphylococcus aureus", 1280),
    ("Pseudomonas aeruginosa", 287),
    ("Streptococcus pneumoniae", 1313),
    ("Enterococcus faecium", 1352),
]
METHODS = ["MIC", "Broth dilution"]
PAGE = 25000  # BV-BRC 单请求上限（实测 limit(25000) 可行，limit(25000,offset) 分页）


def get(url, tries=5):
    last = None
    for a in range(tries):
        try:
            with open_url(url, headers=HEADERS, timeout=300) as r:
                return r.read().decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            last = exc
            if a < tries - 1:
                time.sleep(5 * (a + 1))
    raise RuntimeError(f"GET 失败：{url} -> {last}")


def enc(cons):
    return "&".join(urllib.parse.quote(c, safe="(),.") for c in cons)


def fetch_all(tid, method):
    """按 Content-Range 分页拉完该 (物种, 方法) 的全部行。"""
    cons = [f"eq(taxon_id,{tid})", 'eq(evidence,"Laboratory Method")',
            f'eq(laboratory_typing_method,"{method}")']
    rows, offset = [], 0
    while True:
        page_spec = f"limit({PAGE})" if offset == 0 else f"limit({PAGE},{offset})"
        url = API + "genome_amr/?" + enc(cons) + "&" + page_spec
        last = None
        for a in range(5):
            try:
                with open_url(url, headers=HEADERS, timeout=300) as r:
                    total = None
                    for k, v in r.headers.items():
                        if k.lower() == "content-range":
                            total = int(v.split("/")[-1])
                    body = json.loads(r.read().decode("utf-8"))
                last = None
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(5 * (a + 1))
        if last is not None:
            raise RuntimeError(f"分页失败 offset={offset}：{url} -> {last}")
        rows.extend(body)
        offset += len(body)
        print(f"    {method} offset={offset}/{total}", flush=True)
        if not body or offset >= total:
            break
        time.sleep(0.4)
    return rows


def main():
    man = []
    for name, tid in SPECIES:
        out = DATA / f"mic_{tid}.parquet"
        if out.exists() and out.stat().st_size > 1000:
            print(f"[skip] {name} ({tid}) 已存在")
            man.append([str(out.relative_to(DATA)), str(out.stat().st_size), "skip"])
            continue
        print(f"[fetch] {name} (taxon {tid})")
        recs = []
        for m in METHODS:
            part = fetch_all(tid, m)
            for r in part:
                r["_lab_method"] = m
            recs.extend(part)
            print(f"    {m}: {len(part)} 行")
        df = pd.DataFrame(recs)
        df["species"] = name
        df["taxon_id"] = tid
        df.to_parquet(out, index=False)
        print(f"  -> {out.name}: {len(df)} 行 / {df['genome_id'].nunique()} 基因组")
        man.append([str(out.relative_to(DATA)), str(out.stat().st_size), str(len(df))])

    mp = DATA / "MANIFEST_mic.tsv"
    with mp.open("w", encoding="utf-8-sig", newline="") as f:
        f.write("path\tsize_bytes\trows_or_status\tdate\n")
        for r in man:
            f.write("\t".join(r) + f"\t{DATE}\n")
    print("\n清单:", mp)


if __name__ == "__main__":
    main()
