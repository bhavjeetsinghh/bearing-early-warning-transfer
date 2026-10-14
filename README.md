# Bearing early-warning rules across three run-to-failure datasets

Code and data for the paper *Detection Transfers, False Alarms Do Not: A Label-Free Pooled Evaluation of Bearing Early-Warning Rules on the IMS, XJTU-SY and FEMTO Run-to-Failure Datasets* (Bhavjeet Singh Hora, Florida International University).

The study applies the label-free protocol of the companion IMS study ([ims-bearing-label-free-early-warning](https://github.com/bhavjeetsinghh/ims-bearing-label-free-early-warning)) unchanged to all 36 documented run-to-failure bearings in NASA IMS, XJTU-SY and FEMTO/PRONOSTIA (33 evaluable).

**Main result:** an RMS ratio rule at 1.2× an early-life baseline detects 32 of 33 failures (95% CI 0.84–1.00), but its near-zero false-alarm rate on IMS does not carry over to the other rigs, and no fixed threshold meets a 5% false-alarm target across datasets.

## Protocol fixed in advance
[`PROTOCOL.md`](PROTOCOL.md) was written before any pooled result was computed. Analyses added afterwards (transient vs persistent false alarms, first-alarm timing, cost) are labelled post hoc in the code and the paper.

## Reproduce
```bash
pip install numpy pandas scipy matplotlib
bash run_all.sh
```
Outputs: `results/pooled.json` (all reported numbers), `results/cost_per_bearing.json`, `figures/fig1–3` (PNG and PDF).

## Data
`data/` holds per-snapshot time-domain features (RMS, Pearson kurtosis, skewness, variance, peak-to-peak), not raw signals:

| File | Source |
|---|---|
| `test1.csv`, `test2.csv`, `test3.csv` | NASA IMS bearing data (Lee et al., 2007), features from the companion repository |
| `xjtu_features.csv` | XJTU-SY bearing datasets (Wang et al., 2020), 15 bearings |
| `femto_features.csv` | FEMTO/PRONOSTIA, IEEE PHM 2012 challenge (Nectoux et al., 2012): Learning_set + Full_Test_Set, 17 bearings |

To rebuild the XJTU-SY and FEMTO feature tables from the raw downloads:
```bash
python src/extract_runs.py xjtu  <XJTU-SY folder> data/xjtu_features.csv
python src/extract_runs.py femto <FEMTO folder>   data/femto_features.csv
```

## Files
- `src/unify.py` — common table with clocks (IMS operating hours, XJTU 1 min, FEMTO 10 s)
- `src/pooled.py` — detectors D1–D4, protocol, pooled/per-dataset results, sensitivity, transfer, post hoc analyses
- `src/cost2.py` — renewal-reward cost check with false alarms counted per bearing
- `src/figs2.py` — figures

## License
MIT. The original datasets remain under their providers' terms.
