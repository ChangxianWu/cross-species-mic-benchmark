# Cross-species MIC prediction benchmark

Analysis code accompanying the manuscript:

> **A cross-species benchmark for machine-learning prediction of minimum inhibitory
> concentrations from whole-genome sequences: partial ordering survives, absolute
> calibration does not**
> (manuscript in preparation)

A dry-lab benchmark built entirely from public data: quantitative minimum inhibitory
concentration (MIC) measurements from BV-BRC paired with AMRFinderPlus genotype calls
from NCBI Pathogen Detection, evaluated on a four-level extrapolation ladder (random
split, unseen sequence type, unseen country, unseen species).

**Only the analysis code is provided here.** No data, intermediate caches, result
tables, figures or manuscript files are included; the pipeline downloads what it
needs and writes everything to `data/`, `output/` and `cache/`.

---

## Scripts (run order)

| Script | Purpose |
|---|---|
| `01_fetch_mic.py` | Download quantitative MIC rows from BV-BRC (`genome_amr`, `evidence = Laboratory Method`) for the six species |
| `02_build_dataset.py` | Pair labels with AMRFinderPlus genotype calls on `biosample_accession`; emit the label table, genotype matrix and metadata |
| `03_extrapolation_ladder.py` | Main benchmark: four-level ladder (L1 random / L2 unseen lineage / L3 unseen country / L4 unseen species), 4 model families, 2 feature layers, 146 (species, drug) units |
| `04_calibration_ablation.py` | Calibration slope, split-conformal coverage, feature-layer ablation, censoring and tuning sensitivity, size-matched density decomposition |
| `05_aggregate_results.py` | Rebuild the result tables from the per-unit checkpoints |
| `06_correction_ladder.py` | The same cross-species predictions under four assumptions about the target scale, including the oracle-affine ceiling and k-shot recalibration |
| `07_recalibration_retention.py` | Retention fraction vs within-species performance; sampling strategy for local recalibration; censoring perturbation; discrimination heterogeneity |
| `08_ordering_and_screening.py` | Ordering strength with bootstrap CIs; screening-task precision (flagging the top MIC quartile); head-to-head transfer vs local training at matched label budget |
| `09_label_budget_curve.py` | The budget curve under cross-validation: five label budgets (k=25-400), turning point, hyperparameter symmetry check |

## Quick start

```bash
pip install -r requirements.txt

# 1. Quantitative MIC labels (network access required)
python scripts/01_fetch_mic.py

# 2. Genotype side: place the NCBI Pathogen Detection AMR tables at
#    data/pd/<taxon_id>.amr.metadata.tsv (573, 562, 1280, 287, 1313, 1352),
#    then build the paired dataset:
python scripts/02_build_dataset.py

# 3. Benchmark and sensitivity analyses
python scripts/03_extrapolation_ladder.py all
python scripts/04_calibration_ablation.py all
python scripts/05_aggregate_results.py

# 4. Key analyses behind the manuscript's main claims
python scripts/06_correction_ladder.py
python scripts/07_recalibration_retention.py
python scripts/08_ordering_and_screening.py
python scripts/09_label_budget_curve.py
```

## Data sources

* **Labels**: BV-BRC `genome_amr`, rows with `evidence = "Laboratory Method"`,
  via the public REST API (`www.bv-brc.org/api`). Both `laboratory_typing_method`
  = "MIC" and "Broth dilution" rows are quantitative and are kept.
* **Genotypes**: NCBI Pathogen Detection per-species `*.amr.metadata.tsv`
  (`AMR_genotypes`, `computed_types`), from `ftp.ncbi.nlm.nih.gov/pathogen/Results/`.

Both systems are public and need no registration; please respect their rate limits.

## Two things worth knowing before you reuse this code

1. **The label table carries censoring.** Only about 40% of retained MIC records are
   exact values; the rest are intervals (`<` / `>=`). The pipeline keeps them and flags
   them (the `cens` column) rather than dropping them, and the manuscript reports how
   sensitive the headline numbers are to that choice.
2. **Genome identifiers arrive as strings.** BV-BRC `genome_id` values look numeric
   (`573.66433`) and pandas silently coerces them to floats on a JSON round-trip, which
   turns every feature lookup into a miss and every prediction into the training mean.
   The scripts pin the dtype and assert for missing ids; keep both if you modify them.

## Notes

* Scripts `06`-`09` write intermediate tables into `output/` and read earlier tables
  from it; run them after `05`.
* The random seeds are fixed (and shared across scripts). Changing them changes every
  reported number, so leave them alone if you are reproducing the manuscript.
* Python 3.13; see `requirements.txt` for pinned versions. No GPU is needed - the
  benchmark was designed for a 16 GB laptop.
