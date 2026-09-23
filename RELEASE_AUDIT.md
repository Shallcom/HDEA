# HDEA public-release audit

Release candidate date: 2026-09-23

## Scope

Included:

- frozen Module-I selector and matched controls;
- frozen Module-II residual grouping, posterior scoring, nesting, and fusion;
- restricted-option adapters for the four evaluated answer models;
- portable manifest and paired-statistics scripts;
- frozen configs, aggregate CSVs, and final integrity summaries;
- deterministic unit and regression tests.

Excluded:

- all model weights and benchmark videos;
- all 11 GB of historical experiment artifacts and retired methods;
- feature caches and raw prediction files;
- local absolute paths, credentials, logs, and SSH material.

## Verification

- Public test suite: 11/11 passed.
- Synthetic equivalence audit: Module-I Core and PairwisePointwise outputs
  matched the frozen research implementation on 20 independently seeded
  catalogs, including timestamp-required packets.
- Synthetic equivalence audit: E40/E48 action values, control exclusion,
  ranking, and physical frame sets matched the frozen exact-common-parent
  implementation on 20 independently seeded catalogs.
- Candidate permutation tests passed for both modules.
- E32 subset E40 subset E48 and exact 32/40/48 budgets passed.
- Public-release scan found no model/video/cache artifacts, files over 25 MiB,
  local workspace paths, or credential-like assignments.

The reviewer-facing repository URL is intentionally omitted from this tracked
release during double-blind review. A software licence remains a
repository-owner decision and has not been inferred automatically.
