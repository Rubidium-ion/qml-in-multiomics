# qmlmo — multimodal QML experiment backend

Python backend for running quantum + classical models across imaging, omics and
synthetic data with one interface.

```
ModalityAdapter -> FeatureBlock -> align/clean/normalise -> fuse -> run_cv -> results JSON
```

## Modalities

| key | source | notes |
|---|---|---|
| `imaging_dicom` | DICOM directory, one subdir per subject | CT→HU, PT→SUVbw, MR→z-score. First-order radiomics only (no texture). Needs `pydicom`. |
| `pathology_wsi` | WSI dir, one subdir per slide | `.npy` tiles (numpy only) or pyramidal `.tif`/`.svs` (needs `tifffile`); reads highest level ≤ 2048 px. Z-scored tile statistics + flat-tile fraction. No stain normalisation, no GLCM. |
| `transcriptomics` | TPM/count matrix | log1p + variance filter with **whitelist rescue** |
| `genomics` | mutation/CNV matrix | binarise + min-prevalence filter |
| `metabolomics` | LC-MS/NMR intensities | half-min LOD imputation + log |
| `synthetic` | generator | **known ground truth**, cross-modal interactions |
| `matrix` | .npy/.csv/.tsv | generic fallback |

## Models

`logreg`, `svm_rbf`, `gbdt`, `rf`, `qsvm_fidelity`

## Cohort pointer configs (real data not yet staged)

| config | cohort | host | adapter |
|---|---|---|---|
| `configs/cohort_cptac_ct.yaml` | CPTAC-* CT | TCIA | imaging_dicom |
| `configs/cohort_tcga_tcia_ct.yaml` | TCGA imaging | **TCIA, not GDC** (GDC = omics only) | imaging_dicom |
| `configs/cohort_camelyon17_wsi.yaml` | CAMELYON17 pathology | camelyon17.grand-challenge.org | pathology_wsi |
| `configs/cohort_oasis3_mri.yaml` | OASIS-3 longitudinal MRI | XNAT (DUA required) | imaging_dicom |

Data map + access/staging checklist: `docs/cohorts_imaging.md`.

## CLI

```bash
python -m qmlmo list
python -m qmlmo inspect configs/exp_synthetic_crossmodal.yaml
python -m qmlmo run     configs/exp_synthetic_crossmodal.yaml -o benchmarks/results/x.json
python -m qmlmo sweep   configs/exp_synthetic_crossmodal.yaml --scales pi pi/4 pi/16
```

## The four rules this framework enforces structurally

These are not style preferences. Each one is a bug that already cost us a result.

1. **Encoding scale + kernel off-diagonal mass are always reported.** Every
   quantum run emits them. At scale=π on GDSC our kernel collapsed to off-diagonal
   mass 9.7e-4 — numerically the identity — while still producing AUCs near 0.75.
   Accuracy alone hides this completely.

2. **Feature selection is refit inside every training fold.** Global selection
   inflated a TCGA cohort from AUC 0.438 to 0.971. That is an order of magnitude
   larger than any quantum effect we have ever measured.

3. **Normalisation is fitted on training rows only, per modality block.**
   SUV, Hounsfield units and MR arbitrary units are not commensurable; the DICOM
   adapter refuses to put mixed modalities in one feature space. The pathology
   adapter z-scores per analysis image, making tile features scale-invariant
   across raw .npy values and 0-255 TIFF luminance.

4. **Sample alignment is by explicit ID, never row position.** `fuse()` raises if
   blocks are unaligned rather than silently producing a shifted matrix.

## Degeneracy is two-sided

`kernel_diagnostics` flags `offdiag_mean < 0.01` (kernel → identity, nothing is
similar) **and** `> 0.99` (kernel → all-ones, everything is similar). Both are
unusable, and the second is the one we hit first on synthetic data — the opposite
direction from the GDSC failure.

## Measured on synthetic data (400 samples, 80 features, 3 modalities)

Cross-modal interaction task, encoding-scale sweep:

| scale | off-diag mass | QSVM AUC |
|---|---|---|
| π | 0.099 | **0.696** |
| π/2 | 0.198 | 0.674 |
| π/4 | 0.478 | 0.647 |
| π/8 | 0.783 | 0.616 |
| π/16 | 0.934 | 0.605 |

**The optimum here is π — the opposite of the GDSC cohorts, where π/16 was the
repair and π was degenerate.** The tuned scale is a property of the dataset, not
a constant, which is exactly why it must be reported per experiment rather than
inherited from another paper.

Best classical on the same task: SVM-RBF 0.739. Quantum does not win.

Main-effect negative control (scale=π): QSVM 0.914, logreg 0.954, off-diag 0.115,
no degenerate folds — the quantum arm is healthy and still loses, which is the
correct answer for a task with no interaction structure to exploit.

## Synthetic generator caveat worth knowing

`n_interaction_pairs` defaults to a *small* number deliberately. Summing many
random-sign product terms cancels toward Gaussian noise by CLT: with 27 pairs, a
GBDT given the true informative columns reached AUC 0.538 — indistinguishable
from chance. Every quantum arm would have looked "correctly null" for entirely
the wrong reason. Sparse interactions are also the better model of real epistasis.

## Not yet built

- Real cohort *runs* — pointer configs + adapters exist; data staging is the
  blocker (see `docs/cohorts_imaging.md` for access per cohort)
- Texture radiomics (GLCM/GLRLM) for DICOM and WSI adapters
- Stain normalisation for pathology (matters across CAMELYON17's five centres)
- Id manifests for cross-modal fusion (CPTAC biospecimen↔imaging; TCIA barcode↔GDC case)
- Julia backend — the `Model` interface is the intended boundary
