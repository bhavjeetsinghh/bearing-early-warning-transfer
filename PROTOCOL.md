# Paper 2 analysis protocol (fixed 2026-10-14, before any pooled detection result was computed)

Data: all documented run-to-failure bearings in NASA IMS (4 failures), XJTU-SY (15) and FEMTO/PRONOSTIA (17; Learning_set + Full_Test_Set).
Features per snapshot and axis: RMS, Pearson kurtosis, skewness, crest-factor proxy (as in paper 1).
Clock: IMS operating hours (pauses > 15 min = one 10-min interval); XJTU one snapshot per minute; FEMTO one snapshot per 10 s (snapshot index; timestamps not used).
Invalid trailing snapshots: removed if every axis is below 10% of its median RMS (from the end only).

Windows (causal, in snapshots):
- run-in: snapshots 1-10 discarded
- baseline: snapshots 11-60 (n = 50); all thresholds and T2 parameters estimated here
- monitoring from snapshot 61
Eligibility: a bearing is evaluable only if it has at least 120 valid snapshots (>= 60 monitored). Others are reported but excluded from detection rates.
Scoring windows (post hoc, never used by detectors): healthy-reference zone = snapshot 61 to 85% of life; final 15% of life is ambiguous.
Alarm: k = 3 consecutive exceedances. Sustained warning: start of the alarm block lasting (gaps <= 1% of life) to the end.
Detected: sustained warning >= 1% of life. A bearing is detected if any of its axes is detected (horizontal-only reported as sensitivity).
Detectors: D1 RMS ratio c = 2.0 and c = 1.2 (c = 1.2 is the hypothesis from paper 1); D2 Shewhart mu + 3 sd; D3 Pearson kurtosis > 6; D4 Hotelling T2 alpha = 1e-4.
Outcomes: pooled detection k/n with exact (Clopper-Pearson) 95% CI; per dataset; false-alarm episodes in healthy-reference zones (bearings with >= 1 episode; episodes per 1,000 snapshots); warning time in hours and as % of life.
Sensitivity: final window 10% / 25%; detection threshold 0.5% / 2% of life; k = 1, 5; horizontal axis only.
Transfer: leave-one-dataset-out. On the two training datasets choose the smallest RMS ratio c in {1.10, 1.15, ..., 3.00} for which at most 5% of evaluable bearing-axes raise any false alarm in the healthy-reference zone; report detection and false alarms of that c on the held-out dataset.
