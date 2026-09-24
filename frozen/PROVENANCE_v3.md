# PROVENANCE_v3 --- freeze evidence for the v3.0.0 re-acquisition (REVISION-ID: TSD-20260826)

The frozen reading cards of the re-acquisition are Japanese-language documents in the source
repository; they are not reproduced here. Their identity is fixed by the hashes below, and the
date of the commit that froze each one precedes the measurement it governs.

| document | role | md5 (first 8) | sha256 | last commit (author date) |
|---|---|---|---|---|
| `READCARD_R2_faithfulprose_v2_20260826.md` | READCARD R2 v2 (Arms A/C, ladders; frozen 2026-08-26) | `64b7ea9d` | `ee05b89fb8fc221fcf375ee3095aca60bb6e759b475102e3f9c2a908a4143122` | 701bef444953efc8ec6a2df0b3852d55cc351a2e 2026-08-26T18:22:23+09:00 |
| `READCARD_R2b_coherence_v1_20260826.md` | READCARD R2b (population-null bound; frozen 2026-08-26) | `5980f44c` | `b0a0f16a82ed308dd714f559e1b660764b3aba4a6f49cbfc804236cf12a9651c` | 21590381e575b2ab28f60fd13af3cf986a35fbc2 2026-08-26T18:37:32+09:00 |
| `READCARD_R4_sal_series_v1_20260828.md` | READCARD R4 (order-agreement re-acquisition; frozen 2026-08-28) | `14123f9c` | `66ba17e5fc5175d9dd6db0479170192d8c477c934fc628dc10fe4532e5fc7808` | 881c2f5412a8667ce8d1e00a060223ed8c538120 2026-08-28T10:56:26+09:00 |
| `SPEC_tokenset_acceptance_v1_2.md` | SPEC token-set acceptance v1.2 (also included in the joshaku record) | `e95f4661` | `ea1b4d5f695932e5a1496a37f5907a93e41844c5b64f6a227828f3c48c3b587c` | 2bcdc9ad6f89d42da96915c2ce77a83596cb2d6a 2026-08-27T09:49:25+09:00 |

Instrument (archived): joshaku v1.2.0 (seven modules, combined md5 89b194be...; 20 tests), archived at
doi:10.5281/zenodo.22218669 together with SPEC v1.2 (md5 e95f4661...).
Instrument (as run): the five arm/ladder/null outputs (arm_a_v3, arm_c_v3, cii_l1_v3, cii_l2_v3, cii_nulls_v3)
record in their meta joshaku at combined md5 2bb7fe6c... (source commit ad24326a, 2026-08-26; 19 tests; SPEC v1.1,
md5 85715fe0...), one commit before the v1.2.0 freeze; the states differ only in ranks.py (two alias helpers and their exception class added),
and the modules these runners import (pgrain, masks) are byte-identical, so no recorded value depends on it.
The paper's Revision History (v3.1) states the same.
