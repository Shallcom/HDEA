# Frozen result summary

All values below are recomputed from the exact-common-parent release audit.
VideoMME and LongVideoBench use accuracy; MLVU uses the official unweighted
seven-task M-AVG.

## Qwen3-VL-8B

| Dataset | Uniform32 | HDEA-Core | E40 | E48 | HDEA |
|---|---:|---:|---:|---:|---:|
| VideoMME | 63.59 | 66.56 | 67.44 | 67.52 | **67.44** |
| LongVideoBench | 58.26 | 64.62 | 65.07 | 65.15 | **65.22** |
| MLVU | 63.95 | 73.62 | 73.77 | 74.35 | **74.25** |

## Exact HDEA across answer models

| Backbone | VideoMME | LongVideoBench | MLVU |
|---|---:|---:|---:|
| Qwen3-VL-8B | 67.44 | 65.22 | 74.25 |
| InternVL3-8B | 66.33 | 64.47 | 73.50 |
| mPLUG-Owl3-7B | 55.96 | 58.34 | 65.22 |
| LLaVA-Video-7B | 62.48 | 60.88 | 68.65 |

The exact HDEA system is above matched Uniform32 in all 12 model-dataset
cells. Stage 2 is not monotonically better than every single branch: Full
versus Core has 11 positive and one negative cell, while Full versus E48 has
four positive, one tie, and seven negative cells. The complete CSVs and
cluster-bootstrap summaries are released with the code to preserve these
boundaries rather than report only favorable comparisons.
