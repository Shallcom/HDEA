# Result provenance

The CSV files in this directory are stage-scoped exports of the frozen
aggregate outputs used by the manuscript. Numerical fields are unchanged;
backbone display names follow the manuscript, and Stage-1 exports omit the
superseded pre-alignment Full rows.

- Stage 1 (Broad Competition Coverage) source:
  `paperwork/ICLR2027/experiments/p0/statistics/`
- Stage 2 (Focused Competition Resolution) source:
  `paperwork/ICLR2027/experiments/module2_exact/statistics/`
- Integrity sources: `p0_final_audit.json`, `module2_exact_audit.json`, and
  `module2_selection_final_audit.json`
- Frozen global lock: `paperwork/ICLR2027/FINAL_RESULT_LOCK.sha256`

Large bootstrap replicate arrays, model weights, benchmark videos, feature
caches, and per-question raw predictions are intentionally excluded from the
Git repository. The released summary CSVs retain point estimates, standard
errors, confidence intervals, paired fixes/losses, and exact McNemar results.
Stage-1 release tables contain only selector decomposition and Core results;
all final HDEA results are released exclusively under `results/module2/`.
The full artifacts may be archived separately under dataset-compatible terms.
