# Evaluation History

This file is the canonical record of longitudinal evaluation results for the LAFA pipeline.

Update it after every completed source->future evaluation window. Do not overwrite older results.

## Evaluation protocol

Local development scores are produced with the validated fast scorer:

```bash
uv run python -m scripts.score_eval ...
```

Ground truth must be built with the FunctionBench production protocol:

```text
t0/test_sequences.fasta
t1/test_sequences.fasta
        -> democafa.datacollection.compare_fasta
        -> diff_test_sequences_common.fasta

t0/train_terms_propagated.tsv
t1/train_terms_propagated.tsv
common FASTA
t0/go-basic.obo
t1/go-basic.obo
        -> democafa.groundtruth.classify_ground_truth
        -> groundtruth_NK.tsv
        -> groundtruth_LK.tsv
        -> groundtruth_PK.tsv
        -> groundtruth_PK_known.tsv
        -> groundtruth_targets.tsv
        -> groundtruth_terms_of_interest.txt
```

The local scorer is used for model development and comparison across windows. Final selected models may additionally be checked with the official CAFA evaluator.

## Window status

| Source | Future | Status | Unique proteins | Notes |
|---|---|---:|---:|---|
| Sep_2025 | Nov_2025 | complete | 5,898 | Production GT reproduced exactly |
| Sep_2025 | Dec_2025 | pending | 6018 |  |
| Sep_2025 | Mar_2026 | pending |  |  |
| Nov_2025 | Dec_2025 | pending | 642 |  |
| Nov_2025 | Mar_2026 | pending |  |  |
| Dec_2025 | Mar_2026 | pending |  |  |

---

## Sep_2025 -> Nov_2025

| Method | NK-BPO | NK-CCO | NK-MFO | LK-BPO | LK-CCO | LK-MFO | PK-BPO | PK-CCO | PK-MFO | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hmlp_esm2 | 0.290635 | 0.528824 | 0.665836 | 0.313540 | 0.538106 | 0.640417 | 0.124974 | 0.252771 | 0.192328 | 0.394159 |
| mlp_t5_esm1b | 0.319527 | 0.540823 | 0.652973 | 0.327815 | 0.529418 | 0.727090 | 0.126036 | 0.265901 | 0.207132 | 0.410746 |
| pyb_t5 | 0.292943 | 0.508791 | 0.589232 | 0.294237 | 0.513824 | 0.643089 | 0.122733 | 0.240698 | 0.198052 | 0.378178 |
| blast_knn | 0.034743 | 0.139024 | 0.169080 | 0.061939 | 0.188027 | 0.187016 | 0.025451 | 0.114043 | 0.072295 | 0.110180 |
| naive_prior | 0.001233 | 0.123185 | 0.021401 | 0.001833 | 0.176309 | 0.011559 | 0.001515 | 0.108127 | 0.102133 | 0.060811 |
| nonexp | 0.052236 | 0.164039 | 0.327282 | 0.092178 | 0.182764 | 0.294023 | 0.030656 | 0.108417 | 0.079562 | 0.147906 |
| ltr | 0.321859 | 0.549110 | 0.625878 | 0.337853 | 0.510042 | 0.697163 | 0.127343 | 0.275519 | 0.215626 | 0.406710 |

Source snapshot `Sep_2025`:

- train proteins: 87,925;
- direct protein-GO pairs: 547,694;
- proteins with new evaluable annotations in `Nov_2025`: 5,898.

---

## Sep_2025 -> Dec_2025

| Method | NK-BPO | NK-CCO | NK-MFO | LK-BPO | LK-CCO | LK-MFO | PK-BPO | PK-CCO | PK-MFO | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hmlp_esm2 | 0.293996 | 0.448520 | 0.608897 | 0.313545 | 0.491597 | 0.615149 | 0.128575 | 0.263271 | 0.205829 | 0.374375 |
| mlp_t5_esm1b | 0.321897 | 0.458009 | 0.608207 | 0.330592 | 0.476376 | 0.692372 | 0.128305 | 0.277255 | 0.218789 | 0.390200 |
| pyb_t5 | 0.291478 | 0.443126 | 0.557279 | 0.293716 | 0.485960 | 0.615516 | 0.124848 | 0.247984 | 0.197383 | 0.361921 |
| blast_knn | 0.047090 | 0.146743 | 0.190212 | 0.060733 | 0.187610 | 0.185030 | 0.025683 | 0.090827 | 0.077527 | 0.112384 |
| naive_prior | 0.001523 | 0.122330 | 0.016597 | 0.001711 | 0.171202 | 0.010625 | 0.001729 | 0.091914 | 0.090498 | 0.056459 |
| nonexp | 0.070518 | 0.166544 | 0.422246 | 0.090980 | 0.179575 | 0.287363 | 0.031230 | 0.085967 | 0.095742 | 0.158907 |
| ltr | 0.342977 | 0.466920 | 0.639171 | 0.342334 | 0.477113 | 0.678422 | 0.128311 | 0.253792 | 0.223881 | 0.394769 |

Source snapshot `Sep_2025`:

- train proteins: 87,925;
- direct protein-GO pairs: 547,694;
- proteins with new evaluable annotations in `Dec_2025`: 6018.

---

## Sep_2025 -> Mar_2026

Status: pending.

Results will be appended here after scoring.

---

## Nov_2025 -> Dec_2025

| Method | NK-BPO | NK-CCO | NK-MFO | LK-BPO | LK-CCO | LK-MFO | PK-BPO | PK-CCO | PK-MFO | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hmlp_esm2 | 0.317461 | 0.472419 | 0.511242 | 0.379306 | 0.345886 | 0.396951 | 0.156486 | 0.185733 | 0.322571 | 0.343117 |
| mlp_t5_esm1b | 0.338959 | 0.485447 | 0.562744 | 0.439445 | 0.371130 | 0.438205 | 0.139190 | 0.209805 | 0.359639 | 0.371618 |
| pyb_t5 | 0.311324 | 0.494042 | 0.514703 | 0.336814 | 0.365202 | 0.338946 | 0.143757 | 0.168322 | 0.277550 | 0.327851 |
| blast_knn | 0.102687 | 0.206145 | 0.244406 | 0.065795 | 0.155296 | 0.180545 | 0.029704 | 0.073492 | 0.137544 | 0.132846 |
| naive_prior | 0.002706 | 0.128766 | 0.008480 | 0.002712 | 0.083098 | 0.036188 | 0.004420 | 0.063123 | 0.012923 | 0.038046 |
| nonexp | 0.140474 | 0.167925 | 0.583454 | 0.079474 | 0.119082 | 0.331639 | 0.043264 | 0.092699 | 0.202257 | 0.195585 |
| ltr | 0.472635 | 0.484361 | 0.710073 | 0.452058 | 0.361453 | 0.639389 | 0.141560 | 0.225880 | 0.358427 | 0.427315 |

Source data update from `Sep_2025` to `Nov_2025`:

- train proteins: 87,925 -> 88,068 (+143 net; +360 added, -217 removed);
- direct protein-GO pairs: 547,694 -> 549,522 (+1,828);
- proteins with new evaluable annotations in `Dec_2025`: 642.

---

## Nov_2025 -> Mar_2026

Status: pending.

Results will be appended here after scoring.

---

## Dec_2025 -> Mar_2026

Status: pending.

Results will be appended here after scoring.

---

## Frozen CAFA6 reference

These values are retained only as a historical regression reference. They are not directly comparable to LAFA longitudinal windows.

| Method | NK-BPO | NK-CCO | NK-MFO | LK-BPO | LK-CCO | LK-MFO | PK-BPO | PK-CCO | PK-MFO | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ltr | 0.315207 | 0.557121 | 0.716891 | 0.433852 | 0.566162 | 0.617233 | 0.093376 | 0.264073 | 0.247002 | 0.423435 |
| mlp_t5_esm1b | 0.283562 | 0.523597 | 0.659966 | 0.421434 | 0.533736 | 0.580843 | 0.094824 | 0.238473 | 0.197586 | 0.392669 |
| hmlp_esm2 | 0.296082 | 0.525582 | 0.652158 | 0.411521 | 0.536857 | 0.538785 | 0.090142 | 0.234307 | 0.195645 | 0.386786 |
| pyb_t5 | 0.269920 | 0.493913 | 0.586509 | 0.360221 | 0.495081 | 0.530805 | 0.085600 | 0.223917 | 0.178423 | 0.358266 |
| blast_knn | 0.229281 | 0.458392 | 0.606501 | 0.293237 | 0.490900 | 0.507992 | 0.049579 | 0.203254 | 0.178294 | 0.335270 |
| nonexp | 0.216687 | 0.384937 | 0.512081 | 0.358709 | 0.385568 | 0.500339 | 0.062921 | 0.203975 | 0.188708 | 0.312658 |
| naive_prior | 0.089562 | 0.249441 | 0.093835 | 0.071116 | 0.233640 | 0.075669 | 0.025499 | 0.216515 | 0.030667 | 0.120660 |
