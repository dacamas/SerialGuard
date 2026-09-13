# SerialGuard

### Machine Learning–Based VIN Anomaly and Authenticity Detection

SerialGuard is a complete Google Colab project for studying **structural and statistical anomalies in Vehicle Identification Numbers (VINs)**. It combines a local VIN rule engine, supervised classifiers, normal-only anomaly detectors, and a Gradio interface in one executable notebook.

The central question is whether machine learning adds useful signal beyond explicit VIN rules, especially for subtle transformations that preserve valid formatting and checksums.

**This is an experimental benchmark and analysis demo, not a validated vehicle-authenticity or fraud-detection service.** All training identifiers are generated examples. The recorded results expose substantial false-positive rates rather than establishing operational readiness.

> SerialGuard evaluates VIN structure and statistical patterns. It does not determine whether a vehicle is stolen, cloned, counterfeit, legally registered, or otherwise fraudulent.

## Run the project

The complete implementation is in [SerialGuard.ipynb](SerialGuard.ipynb).

1. Download the notebook from this repository.
2. Open Google Colab and upload the `.ipynb` file.
3. Select **Runtime → Run all**.
4. Review the generated analyses and use the Gradio interface at the end.

The notebook installs its dependencies, retrieves public reference data, constructs the benchmark, trains and evaluates every model, saves artifacts, and launches the demo. **No dataset upload, API key, Kaggle token, Google Drive mount, or paid data service is required.** An internet connection is needed for package installation and live downloads. CPU execution is supported; the autoencoder uses a GPU if one is available.

For a smaller run, change `QUICK_MODE = True` in the configuration cell before executing. No configuration changes are required for the default experiment.

## What the notebook includes

- Automated NHTSA reference ingestion with integrity checks, caching, XML fallback, and an attributed bundled snapshot.
- VIN normalization, checksum calculation, structural validation, and sanity tests.
- Sixteen synthetic corruption types with source-group IDs and edit metadata.
- Source-group splitting and assertions against group overlap and exact-string leakage.
- Interpretable structural features, positional one-hot encoding, and character TF-IDF n-grams.
- Logistic Regression, Random Forest, XGBoost, Isolation Forest, One-Class SVM, and a PyTorch autoencoder.
- Validation-only tuning, probability calibration, threshold selection, and a locked final test evaluation.
- Easy/hard evaluations, unseen-category experiments, ablations, severity analysis, and distribution-shift diagnostics.
- Feature importance, logistic contributions, masked error examples, and a hybrid Gradio analysis interface.

```text
Public NHTSA WMI reference
          ↓
Reference cleaning → generated source VINs
          ↓
Split source groups → generate descendants within each partition
          ↓
Training-only preprocessing
          ↓
Rules / supervised models / normal-only detectors
          ↓
Validation tuning, calibration and threshold lock
          ↓
Final test evaluation and fixed diagnostic experiments
          ↓
Hybrid SerialGuard engine → Gradio
```

## Data provenance and benchmark scope

The notebook uses the **NHTSA Product Information Catalog and Vehicle Listing (vPIC)** WMI reference API:

- [Honda name-query endpoint](https://vpic.nhtsa.dot.gov/api/vehicles/GetWMIsForManufacturer/honda?vehicleType=2&format=json)
- [Ford name-query endpoint](https://vpic.nhtsa.dot.gov/api/vehicles/GetWMIsForManufacturer/ford?vehicleType=2&format=json)
- [NHTSA vPIC API documentation](https://vpic.nhtsa.dot.gov/api/)

These are bounded manufacturer-reference queries, not individual VIN-decoding requests. Manufacturer names are matched partially, so the returned records are not a manually curated list containing only Honda and Ford entities.

The saved run retrieved both sources live on **September 13, 2026**, obtaining **32 reference records and 31 usable WMIs** after excluding one malformed or out-of-scope record. Retrieval times, source modes, URLs, and hashes are recorded by the notebook. If live downloads fail, validated cached data or the bundled snapshot can be used; their provenance remains explicit.

**The reference data does not supply an observed vehicle-VIN training cohort.** The notebook therefore distinguishes three kinds of data:

| Data | Origin and meaning |
| --- | --- |
| WMI reference records | Public NHTSA manufacturer/reference information. |
| Generated normal VINs | Synthetic strings that pass the implemented rule subset; not observed or confirmed manufacturer-assigned VINs. |
| Synthetic anomalies | Controlled transformations of generated source VINs; a transformation label does not establish real-world invalidity or fraud. |

For each WMI, the generator creates six simulated descriptor templates and two simulated plant symbols. It samples year codes for 2010–2026, six-digit production sequences, and a declared synthetic WMI frequency distribution. **Descriptor, plant, and sampling patterns are experimental assumptions, not verified manufacturer specifications.**

The local validator implements selected U.S.-market, high-volume passenger-car rules: 17-character length, the permitted character set, checksum consistency, model-year code validity, and a numeric final-five-character suffix. Unknown WMIs produce a limited-coverage warning. Low-volume and non-U.S. cases require additional context.

Rule basis: [49 CFR Part 565, official 2023 annual edition, §§565.13–565.15](https://www.govinfo.gov/content/pkg/CFR-2023-title49-vol6/pdf/CFR-2023-title49-vol6-part565-subpartB.pdf). This implementation is a documented subset, not exhaustive regulatory certification.

## Experimental design

### Source groups and leakage prevention

The default benchmark contains **2,400 source groups**, each with one generated normal and three corrupted descendants: **9,600 rows, including 7,200 anomalies and zero observed vehicle VINs**.

Sources are normalized and deduplicated before splitting. Each source and every descendant remain in the same partition. Exact collisions with source VINs or previously generated descendants are rejected.

| Partition | Source groups | Rows | Purpose |
| --- | ---: | ---: | --- |
| Training | 1,680 | 6,720 | Preprocessing and model fitting |
| Calibration validation | 180 | 720 | Supervised calibration and normal-only autoencoder monitoring |
| Selection validation | 180 | 720 | Hyperparameters, retained calibration policy, model selection, and thresholds |
| Final test | 360 | 1,440 | Locked evaluation |
