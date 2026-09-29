# ML4HL Responsible AI Toolbox: HIV Treatment Failure Risk Stratification

This project provides a workflow for **retrospective HIV treatment failure risk stratification** using baseline patient characteristics from the AIDS Clinical Trials Group Study 175 (ACTG 175) and the Microsoft Responsible AI (RAI) Toolbox. The binary target `cid` records whether a patient experienced treatment failure (disease progression or death) during the observation period.

> **Disclaimer:** This workflow is an assignment exercise using a 1990s RCT dataset. It is **not validated** for prospective prediction, clinical deployment, or regulatory submission. Modern HIV care uses different treatment regimens (HAART/ART) that render this model clinically obsolete without revalidation.

---

## Introduction and Purpose

- **Motivation:** Identifying HIV-positive patients at high risk of treatment failure enables proactive clinical intervention — intensified monitoring, adherence support, or treatment switch — before disease progression occurs.
- **Goal:** Build a transparent, calibrated classification model and use Responsible AI tools to audit its behaviour, identify deployment risks (particularly racial and gender disparities), and propose clinician-ready mitigations.
- **Domain context:** The model serves as a *risk-stratification flag* (not a diagnosis). A positive flag triggers closer monitoring. Missing a true treatment failure (FN) is clinically more costly than an unnecessary monitoring visit (FP).
- **Unique advantage:** ACTG 175 is a **randomised clinical trial** (Hammer et al., *NEJM* 1996), making treatment assignment independent of confounders. This enables **credible causal analysis** — a rare opportunity in healthcare ML.

Adapted from: [IE-ML-for-Healthcare/RAI_opioid_risk_prevention](https://github.com/IE-ML-for-Healthcare/RAI_opioid_risk_prevention)

---

## Project Structure

| File | Role |
|---|---|
| `ML4HL_ACTG175_RAI_toolbox.ipynb` | Main notebook — all 9 rubric steps |
| `utils.py` | Helper functions: subgroup reports with Wilson CIs, calibration (ECE, reliability), bootstrap CIs, threshold policies and stability, tiered alerts, group thresholds, leakage audit, error-tree paths, what-if by arm, monitoring check |
| `figures/` | Seven PNGs written by the notebook (data overview, coefficients, reliability, trade-off frontier, error heat-map, feature importance, causal) |
| `ML4HL_ACTG175_RAI_presentation.pptx` | 14-slide RAI toolkit deck (dashboard screenshots, insights, mitigations, policy pros/cons/consequences) |
| `data/ACTG175.csv` | AIDS Clinical Trials Group Study 175 dataset (2,139 records) |
| `environment.yml` | Conda environment with pinned dependencies |
| `requirements.txt` | pip requirements (alternative to conda) |
| `SETUP_GUIDE.md` | Step-by-step local setup with troubleshooting |
| `README.md` | This file |
| `LICENSE` | MIT license for the code (the dataset keeps its own UCI terms) |

---

## Install and Environment

### Prerequisites
- Python 3.10.x (required — RAI packages do not support 3.12+)
- Conda/Mamba or pip + venv
- Git

### Option A — pip + venv (Recommended)
```bash
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Option B — Conda/Mamba
```bash
conda env create -f environment.yml
conda activate actg175_rai
```

### Verify
```bash
python -c "import sklearn, responsibleai, fairlearn, econml; print('✅ All OK')"
```

### Key dependencies
- Python 3.10, scikit-learn 1.5.1, pandas 1.5.3, numpy 1.26.2, matplotlib 3.9.1, seaborn 0.13.2
- Full tested versions (interpret-community 0.32.0, erroranalysis 0.5.5, dice-ml 0.11, raiutils 0.4.2 ...) are pinned in `requirements.txt` and `environment.yml`
- responsibleai 0.34.0, raiwidgets 0.34.0 (RAI Dashboard)
- fairlearn (fairness metrics; version resolved by pip, 0.7.0 in the tested environment)
- econml 0.15.1 (causal inference)
- ucimlrepo 0.0.7 (dataset download; optional, the CSV is included in `data/`)

---

## Quickstart Workflow

```bash
# 1. Activate environment
source .venv/bin/activate   # or: conda activate actg175_rai

# 2. Launch Jupyter
jupyter lab

# 3. Open ML4HL_ACTG175_RAI_toolbox.ipynb, select the `actg175_rai` kernel and Run All (about 10-15 minutes; the RAI step is the slowest)

If the kernel is missing: `python -m ipykernel install --user --name actg175_rai`.
```

The notebook will:
1. Load the ACTG 175 dataset (from local CSV or UCI download fallback)
2. Exclude post-randomization features to prevent temporal leakage
3. Create ONE treatment column `arm` (ZDV, ZDV+ddI, ZDV+Zal, ddI), one-hot encoded inside the pipeline
4. Split data 70/15/15 (train / validation / test) with stratification
5. Build a preprocessing + logistic regression pipeline (no leakage)
6. Evaluate discrimination (ROC AUC, PR AUC) with subgroup analysis by race and gender
7. Calibrate probabilities and assess with reliability plots and Brier score
8. Compare threshold policies, show that the joint target is infeasible, and lock a two-tier operating point on validation
9. Report final test performance with bootstrap CIs and a 20-seed re-split stability check
10. (Step 9) Launch the Responsible AI Dashboard for interpretability, error analysis, counterfactuals, and causal inference

---

## Data Description

- **Source:** [UCI ML Repository — AIDS Clinical Trials Group Study 175](https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175)
- **Publication:** Hammer SM et al. A Trial Comparing Nucleoside Monotherapy with Combination Therapy in HIV-Infected Adults with CD4 Cell Counts from 200 to 500 per Cubic Millimeter. *NEJM* 1996;335:1081-90.
- **Records:** 2,139 HIV-positive adults
- **Target:** `cid` — binary (1 = treatment failure / death, 0 = censored / survived)
- **Missing values:** None

### Features Used as Predictors

| Feature | Description | Clinical Relevance |
|---|---|---|
| `age` | Age (years) | Older age → different immune recovery |
| `wtkg` | Weight (kg) | Wasting = advanced disease |
| `karnof` | Karnofsky performance score (0–100) | Functional status; lower = sicker |
| `preanti` | Months of prior antiretroviral therapy | Treatment experience |
| `cd40` | CD4 count at baseline (cells/mm³) | **Key immune marker** — lower = more immunocompromised |
| `cd80` | CD8 count at baseline (cells/mm³) | Immune activation marker |
| `hemo` | Hemophilia (0/1) | Transmission route |
| `homo` | Homosexual activity (0/1) | Transmission route |
| `drugs` | IV drug use history (0/1) | Transmission route, comorbidity risk |
| `oprior` | Non-ZDV ARV prior to study (0/1) | Prior treatment exposure |
| `z30` | ZDV in 30 days prior (0/1) | Recent ZDV exposure |
| `zprior` | ZDV prior to study (0/1) | Prior ZDV history |
| `race` | Race (0=White, 1=Non-white) | **Sensitive attribute** for fairness |
| `gender` | Gender (0=Female, 1=Male) | **Sensitive attribute** for fairness |
| `str2` | Antiretroviral history (0=naive, 1=experienced) | Stratification variable |
| `symptom` | Symptomatic indicator (0/1) | Disease stage |
| `arm` | Treatment arm, one column (ZDV is the reference; one-hot inside the pipeline) | **Treatment assignment** |

### Features EXCLUDED (Leakage Prevention)

| Feature | Reason |
|---|---|
| `cd420` | CD4 at 20 weeks — post-randomization (temporal leakage) |
| `cd820` | CD8 at 20 weeks — post-randomization (temporal leakage) |
| `offtrt` | Off-treatment indicator — observed during trial (temporal leakage) |
| `time` | Time to failure/censoring — direct target leakage |
| `trt` | Replaced by the single `arm` column |
| `treat` | Deterministic function of `arm` (any therapy vs ZDV) — redundant |
| `strat` | Stratification code — redundant with `str2` + `preanti` |
| `pidnum` | Patient ID — absent from the supplied CSV; would be excluded |

### Treatment Arms

| Arm | Code | Regimen | Type |
|---|---|---|---|
| 0 | ZDV | Zidovudine only | Monotherapy (reference) |
| 1 | ZDV+ddI | Zidovudine + didanosine | Combination |
| 2 | ZDV+Zal | Zidovudine + zalcitabine | Combination |
| 3 | ddI | Didanosine only | Alternative monotherapy |

### CD4/CD8 Clinical Relevance
- **CD4 count** is the primary measure of immune function in HIV. Normal: 500–1500/mm³. The trial enrolled patients with 200–500/mm³ (moderate immunosuppression). Lower baseline CD4 → higher risk of disease progression.
- **CD8 count** reflects immune activation. Elevated CD8 can indicate active viral replication or immune response.

---

## Evaluation and Thresholding

### Metrics
- **ROC AUC** and **PR AUC** for discrimination
- **Brier score** and reliability plots for calibration
- **Recall, Precision, Confusion matrix** at the chosen threshold
- Subgroup metrics by **race** and **gender** for fairness

### Threshold Policies Compared
1. **Recall-floor**: recall ≥ 60%, then maximise precision (thr 0.279, ~371 alerts/1,000 on validation)
2. **Workload-cap**: ≤ 300 alerts per 1,000, then maximise true positives (thr 0.312, recall ~54%)
3. **Cost-sensitive**: illustrative harm weights (FN cost = 10× FP cost); flags almost everyone
4. **Joint**: recall ≥ 60% AND alerts ≤ 300/1,000 — **infeasible** (60% recall needs at least 358 alerts/1,000)

**Locked operating point (validation, before touching test):** a two-tier plan. Tier 1 (score ≥ 0.312, about 290/1,000) gets intensive follow-up; Tier 2 (0.279–0.312, about 81/1,000) gets a light-touch check. The notebook reports how the target trade-off fails rather than hiding it.

### Trade-off Rationale
In HIV treatment monitoring, missing a patient who is failing therapy (FN) has far greater consequences than an unnecessary monitoring visit (FP). The two-tier plan keeps intensive workload near capacity while the lighter tier recovers extra recall.

### Executed results (seed 42; test set 321 patients, 78 failures)

| Quantity | Value |
|---|---|
| Validation ROC AUC | 0.731 |
| Test ROC AUC | 0.660 (95% CI 0.594–0.730) |
| Test recall at locked threshold | 0.397 (CI 0.30–0.51) |
| Test alerts per 1,000 | 265 |
| 20 random re-splits, test recall | 0.62 ± 0.12 at about 403 alerts/1,000 |

With about 78 test events, single-split numbers are noisy; treat the 20-split figures as the honest expectation.

---

## Responsible AI and Reproducibility

### RAI Dashboard Components
- **Interpretability:** Global and local feature importance — which baseline features drive failure risk scores
- **Error Analysis:** Heatmaps by race × treatment arm to identify high-error segments
- **Counterfactuals:** "What if this patient had received combination therapy?" — directly actionable because treatment is a modifiable feature
- **Causal Inference:** Treatment effect estimates using RCT data — **credible causal analysis** because treatment was randomised

### Causal Analysis — RCT Advantage
This is randomised clinical trial data, making causal estimates more credible than observational studies. The causal component estimates:
- **Average Treatment Effect (ATE):** Does combination therapy reduce failure risk vs. monotherapy?
- **Heterogeneous Treatment Effects (HTE):** Which patients benefit most from combination therapy? (e.g., those with low baseline CD4)

### Reproducibility
- Random seed: `RANDOM_STATE = 42` used throughout (splits, CV, calibration, models, EconML forests, ThresholdOptimizer); NumPy is also seeded
- Data split: 70/15/15 (1,497/321/321), stratified on outcome × arm
- All preprocessing fitted on training data only (inside `sklearn.Pipeline`)
- Calibration performed on training folds only (5-fold CV, sigmoid); raw, sigmoid and isotonic are compared on validation
- Threshold locked on validation before test evaluation
- Environment captured with tested versions in `environment.yml` and `requirements.txt`
- Notebook is committed with executed outputs so results can be checked without re-running
- The RAI causal wrapper is not seeded, so its numbers vary slightly between runs. The notebook therefore also reports a seeded direct EconML estimate and the raw RCT risk differences (headline causal figures)

### Governance
- This is an exercise using publicly available RCT data
- In clinical settings: ensure IRB approval, patient consent, bias auditing, and regulatory alignment
- Monitor recall, precision, and calibration monthly by race and gender
- Alert if recall drops below 50% in any subgroup

---

## Troubleshooting

| Issue | Fix |
|---|---|
| RAI dashboard not rendering | Trust notebook (File → Trust), use JupyterLab ≥ 3.x |
| Import errors | Recreate env: `conda env remove -n actg175_rai && conda env create -f environment.yml` |
| Plots not showing | Ensure `%matplotlib inline` or use JupyterLab default |
| Widget errors | Verify: `python -c "import ipywidgets; print(ipywidgets.__version__)"` |
| Dataset not found | The CSV is in `data/ACTG175.csv`; if missing the notebook downloads from UCI (needs `ucimlrepo` and internet) |
| Kernel `actg175_rai` missing | `python -m ipykernel install --user --name actg175_rai` |
| Dashboard port busy | Pass a different `port=` to `ResponsibleAIDashboard` in Step 9 |
| Causal step prints an EconML covariance warning | Known in the RAI wrapper; use the seeded EconML and raw RCT numbers shown next to it |
| Python 3.12+ errors | RAI packages require Python 3.10. Use `python3.10 -m venv .venv` |

---

## License and Acknowledgements

- **License:** MIT for the code in this repository (see `LICENSE`). The ACTG 175 data is distributed by UCI under CC BY 4.0; cite Hammer et al. 1996.
- **Dataset:** [UCI ML Repository — ACTG 175](https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175) by Hammer et al.
- **Publication:** Hammer SM et al. *NEJM* 1996;335:1081-90.
- **Reference workflow:** [IE-ML-for-Healthcare/RAI_opioid_risk_prevention](https://github.com/IE-ML-for-Healthcare/RAI_opioid_risk_prevention)
- **Tools:** Microsoft Responsible AI Toolbox, scikit-learn, Fairlearn, EconML
