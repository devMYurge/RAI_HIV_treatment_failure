# ML4HL Responsible AI Toolbox: HIV Treatment Failure Risk Stratification

This project provides a workflow for **retrospective HIV treatment failure risk stratification** using baseline patient characteristics from the AIDS Clinical Trials Group Study 175 (ACTG 175) and the Microsoft Responsible AI (RAI) Toolbox. The binary target `cid` records whether a patient experienced treatment failure (disease progression or death) during the observation period.

> **Disclaimer:** This workflow is a assignment exercise using a 1990s RCT dataset. It is **not validated** for prospective prediction, clinical deployment, or regulatory submission. Modern HIV care uses different treatment regimens (HAART/ART) that render this model clinically obsolete without revalidation.

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
| `utils.py` | Helper functions: AUC reporting, threshold policies, plots |
| `data/ACTG175.csv` | AIDS Clinical Trials Group Study 175 dataset (2,139 records) |
| `environment.yml` | Conda environment with pinned dependencies |
| `requirements.txt` | pip requirements (alternative to conda) |
| `SETUP_GUIDE.md` | Local environment setup with troubleshooting |
| `README.md` | This file |
| `LICENSE` | Repository license |

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
- Python 3.10, scikit-learn 1.5.1, pandas, numpy, matplotlib, seaborn
- responsibleai 0.34.0, raiwidgets 0.34.0 (RAI Dashboard)
- fairlearn 0.10.0 (fairness metrics)
- econml 0.15.1 (causal inference)
- ucimlrepo (dataset download — optional)

---

## Quickstart Workflow

```bash
# 1. Activate environment
source .venv/bin/activate   # or: conda activate actg175_rai

# 2. Launch Jupyter
jupyter lab

# 3. Open ML4HL_ACTG175_RAI_toolbox.ipynb and Run All
```

The notebook will:
1. Load the ACTG 175 dataset (from local CSV or UCI download fallback)
2. Exclude post-randomization features to prevent temporal leakage
3. Create binary treatment indicator and treatment arm dummies
4. Split data 70/15/15 (train / validation / test) with stratification
5. Build a preprocessing + logistic regression pipeline (no leakage)
6. Evaluate discrimination (ROC AUC, PR AUC) with subgroup analysis by race and gender
7. Calibrate probabilities and assess with reliability plots and Brier score
8. Compare four threshold selection policies and lock a final threshold
9. Report final test performance with confidence intervals
10. Launch the Responsible AI Dashboard for interpretability, error analysis, counterfactuals, and causal inference

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
| `trt_1/2/3` | Treatment arm dummies (ref=ZDV only) | **Treatment assignment** |

### Features EXCLUDED (Leakage Prevention)

| Feature | Reason |
|---|---|
| `cd420` | CD4 at 20 weeks — post-randomization (temporal leakage) |
| `cd820` | CD8 at 20 weeks — post-randomization (temporal leakage) |
| `offtrt` | Off-treatment indicator — observed during trial (temporal leakage) |
| `time` | Time to failure/censoring — direct target leakage |
| `pidnum` | Patient ID — not a feature |
| `strat` | Stratification code — redundant with `str2` + `preanti` |

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
1. **Recall-floor**: recall ≥ 60%, then maximise precision
2. **Workload-cap**: ≤ 300 alerts per 1,000, then maximise true positives
3. **Cost-sensitive**: illustrative harm weights (FN cost = 10× FP cost)
4. **Joint**: recall ≥ 60% AND alerts ≤ 300/1,000 — the final locked threshold

### Trade-off Rationale
In HIV treatment monitoring, missing a patient who is failing therapy (FN) has far greater consequences than an unnecessary monitoring visit (FP). The threshold prioritises recall while keeping alert volume manageable for clinical teams.

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
- Random seed: `RANDOM_STATE = 42` used throughout
- All preprocessing fitted on training data only (inside `sklearn.Pipeline`)
- Calibration performed on training folds only (5-fold CV)
- Threshold locked on validation before test evaluation
- Environment captured in `environment.yml` and `requirements.txt`

### Governance
- This is a exercise using publicly available RCT data
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
| Dataset not found | Notebook auto-downloads from UCI; or manually place CSV in `data/ACTG175.csv` |
| Python 3.12+ errors | RAI packages require Python 3.10. Use `python3.10 -m venv .venv` |

---

## License and Acknowledgements

- **License:** See `LICENSE`
- **Dataset:** [UCI ML Repository — ACTG 175](https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175) by Hammer et al.
- **Publication:** Hammer SM et al. *NEJM* 1996;335:1081-90.
- **Reference workflow:** [IE-ML-for-Healthcare/RAI_opioid_risk_prevention](https://github.com/IE-ML-for-Healthcare/RAI_opioid_risk_prevention)
- **Tools:** Microsoft Responsible AI Toolbox, scikit-learn, Fairlearn, EconML
