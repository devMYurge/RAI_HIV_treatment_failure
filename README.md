# ML4HL Responsible AI Toolbox: Hepatitis C Liver Disease Risk Stratification

This project provides a teaching workflow for **retrospective liver disease risk stratification** using routine blood-test biomarkers and the Microsoft Responsible AI (RAI) Toolbox. The binary target `LiverDisease` records whether a patient has documented Hepatitis, Fibrosis, or Cirrhosis versus being a healthy blood donor.

> **Disclaimer:** This workflow is a classroom exercise. It is **not validated** for prospective diagnosis, clinical deployment, or regulatory submission.

---

## Introduction and Purpose

- **Motivation:** Hepatitis C is a leading cause of chronic liver disease worldwide. Many patients remain asymptomatic until late-stage fibrosis or cirrhosis. Early detection via routine laboratory screening could substantially reduce diagnostic delay.
- **Goal:** Build a transparent, calibrated classification model and use Responsible AI tools to audit its behaviour, identify deployment risks, and propose clinician-ready mitigations.
- **Domain context:** The model serves as a *screening flag* (not a diagnosis). A positive flag triggers confirmatory workup (e.g., HCV-RNA test, liver biopsy). Missing a true case (false negative) is clinically more costly than an unnecessary follow-up (false positive).

Adapted from: [IE-ML-for-Healthcare/RAI_opioid_risk_prevention](https://github.com/IE-ML-for-Healthcare/RAI_opioid_risk_prevention)

---

## Project Structure

| File | Role |
|---|---|
| `ML4HL_HepC_RAI_toolbox.ipynb` | Main notebook — all 9 rubric steps |
| `utils.py` | Helper functions: AUC reporting, threshold policies, plots |
| `Data/HepatitisCdata.csv` | UCI Hepatitis C dataset (615 records, 14 attributes) |
| `environment.yml` | Conda environment with pinned dependencies |
| `README.md` | This file |
| `LICENSE` | Repository license |

---

## Install and Environment

### Prerequisites
- Conda or Mamba, Git, and a Jupyter-compatible environment

### Setup
```bash
# Create the environment
conda env create -f environment.yml
conda activate hepc_rai

# Verify
python -c "import sklearn, responsibleai; print('OK')"
```

### Key dependencies
- Python 3.10+
- scikit-learn, pandas, numpy, matplotlib, seaborn
- responsibleai, raiwidgets (for the RAI Dashboard)
- fairlearn (for fairness metrics)

---

## Quickstart Workflow

```bash
# 1. Activate environment
conda activate hepc_rai

# 2. Launch Jupyter
jupyter lab

# 3. Open ML4HL_HepC_RAI_toolbox.ipynb and run all cells top-to-bottom
```

The notebook will:
1. Load and clean the Hepatitis C dataset
2. Engineer a binary target (liver disease vs healthy)
3. Split data 70/15/15 (train / validation / test) with stratification
4. Build a preprocessing + logistic regression pipeline (no leakage)
5. Evaluate discrimination (ROC AUC, PR AUC) and subgroup metrics
6. Calibrate probabilities and assess with reliability plots and Brier score
7. Compare threshold selection policies and lock a final threshold
8. Report final test performance with confidence intervals
9. Launch the Responsible AI Dashboard for interpretability, error analysis, counterfactuals, and causal inference

---

## Data Description

- **Source:** [UCI Machine Learning Repository — HCV Data](https://archive.ics.uci.edu/dataset/571/hcv+data)
- **Records:** 615 patients
- **Target:** `LiverDisease` — binary (1 = Hepatitis/Fibrosis/Cirrhosis, 0 = Blood Donor/Suspect Donor)
- **Prevalence:** ~12.2% (75 disease cases)

### Features

| Feature | Description | Clinical relevance |
|---|---|---|
| Age | Patient age (years) | Older age → higher HCV prevalence |
| Sex | Biological sex (m/f) | Sex differences in disease progression |
| ALB | Albumin (g/L) | ↓ in liver disease (synthetic failure) |
| ALP | Alkaline phosphatase (IU/L) | ↑ in cholestatic patterns |
| ALT | Alanine aminotransferase (IU/L) | ↑ = active hepatocyte damage |
| AST | Aspartate aminotransferase (IU/L) | ↑ = hepatocyte/muscle damage |
| BIL | Bilirubin (µmol/L) | ↑ = impaired conjugation/excretion |
| CHE | Cholinesterase (kU/L) | ↓ in severe liver disease |
| CHOL | Cholesterol (mmol/L) | May ↓ in cirrhosis |
| CREA | Creatinine (µmol/L) | ↑ in hepatorenal syndrome |
| GGT | Gamma-glutamyl transferase (IU/L) | ↑ in liver/biliary disease |
| PROT | Total protein (g/L) | Reflects liver synthetic function |

### Caveats
- Single-centre dataset with limited disease cases
- No temporal information — retrospective stratification only
- "Suspect blood donor" category conservatively grouped with healthy donors
- Missing values exist for ALP (18), CHOL (10), ALB (1), ALT (1), PROT (1) — handled by median imputation inside the pipeline

---

## Evaluation and Thresholding

### Metrics
- **ROC AUC** and **PR AUC** for discrimination
- **Brier score** and reliability plots for calibration
- **Recall, Precision, Confusion matrix** at the chosen threshold
- Subgroup metrics by sex for fairness

### Threshold policies compared
1. **Recall-floor**: recall ≥ 60%, then maximize precision
2. **Workload-cap**: ≤ 300 alerts per 1,000, then maximize true positives
3. **Cost-sensitive**: illustrative harm weights (FN cost = 10× FP cost)
4. **Joint**: recall ≥ 60% AND alerts ≤ 300/1,000 — the final locked threshold

### Trade-off rationale
In a screening context, missing a liver disease case (FN) has far greater clinical consequences than an unnecessary follow-up test (FP). The threshold is chosen to prioritize recall while keeping alert volume manageable for follow-up clinics.

---

## Responsible AI and Reproducibility

### RAI Dashboard components
- **Interpretability:** global and local feature importance — which lab values drive risk scores
- **Error Analysis:** heatmaps and decision trees identifying high-error patient segments
- **Counterfactuals:** individual "what-if" scenarios for clinician-ready explanations
- **Causal Inference:** exploratory uplift estimates for candidate interventions

### Reproducibility
- Random seed: `RANDOM_STATE = 42` used throughout
- All preprocessing fitted on training data only (inside `sklearn.Pipeline`)
- Threshold locked on validation before test evaluation
- Environment captured in `environment.yml`

### Governance
- This is a teaching exercise using publicly available data
- In clinical settings: ensure IRB approval, patient consent, bias auditing, and regulatory alignment
- Monitor recall, precision, and calibration monthly in deployment; alert if recall drops below 50% in any subgroup

---

## Troubleshooting

| Issue | Fix |
|---|---|
| RAI dashboard not rendering | Trust notebook (File → Trust), use JupyterLab ≥ 3.x |
| Import errors | Recreate env: `conda env remove -n hepc_rai && conda env create -f environment.yml` |
| Plots not showing | Ensure `%matplotlib inline` or use JupyterLab default |
| Widget errors | Verify: `python -c "import ipywidgets; print(ipywidgets.__version__)"` |

---

## License and Acknowledgements

- **License:** See `LICENSE`
- **Dataset:** [UCI ML Repository — HCV Data](https://archive.ics.uci.edu/dataset/571/hcv+data) by Lichtinghagen et al.
- **Reference workflow:** [IE-ML-for-Healthcare/RAI_opioid_risk_prevention](https://github.com/IE-ML-for-Healthcare/RAI_opioid_risk_prevention)
- **Tools:** Microsoft Responsible AI Toolbox, scikit-learn, Fairlearn
