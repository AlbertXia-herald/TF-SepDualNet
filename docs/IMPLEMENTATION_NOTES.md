# Implementation and result provenance

## Scope of this revision

This is the first presentation update, prepared against commit `c729fed58be411139d3a75a9946145735f783b56`.

- Rewrite the repository homepage around TF-SepDualNet.
- Include the original SURF-2025-0528 PDF and faithful rendered previews.
- Preserve the former README as upstream-oriented reference material.
- Correct two misleading source comments without changing executable Python.

No training, evaluation, or log reanalysis was performed. No architecture, loss, optimizer, seed, configuration, manifest, checkpoint, or historical metric was changed.

## Poster provenance

The original file is `SURF-2025-0528.pdf`, titled *Data-Efficient and Low-Complexity Acoustic Scene Classification*. It is a team poster, not a standalone TF-SepDualNet paper.

The README table reproduces Figure 4 as printed. Figure 5 states that starred results correspond to the city-label experiment, while Figure 4 uses a seen/unseen-device heading. This ambiguity is recorded rather than resolved by guessing. No new performance delta or claim of a matched ablation is computed.

The full PDF is included unchanged. The images are page/region renders, not redrawn charts. Preview pixels are not a replacement for the original PDF. The original Figure 5 text refers to an independent test set; that wording remains part of the historical source and is not newly verified by this update.

The lightweight-model section's `student_KD` figures describe another branch of the team project. They must not be used as this model's parameter count, compute cost, or accuracy.

## Evaluation provenance

At the reviewed repository commit, `data/meta_dcase_2022/valid.csv` and `test.csv` share blob SHA `389d2811661130f0f09e5c66381db56d1784946c`. They are byte-identical. This observation is about the current checked-in manifests; it does not establish which manifests were used in the original poster runs.

The TF-SepDualNet YAML files select `target_set: total`; the loader randomly pairs source recordings with target-pool samples. Sample overlap and the intended inductive/transductive protocol remain to be verified when historical run records or data become available. No split is changed in this revision.

Until those records are checked, use “reported in the 2025 poster” for the table rather than “independently reproduced” or “verified held-out generalization.”

## Implementation interpretation

- `TFSepDualNet` has two scene heads and two domain heads.
- Validation/test inference averages the scene logits.
- Both device and city domain losses are computed in the current `training_step`.
- The consistency loss is L1 distance between target-head logits. Despite the previous comment, the returned head outputs have not been softmax-normalized.
- The group-validation callback always reads device labels, including when the configuration sets `domain_label: city`.
- Its configured source/target groups `[0,1,2]` and `[3,4,5]` do not cover all nine device IDs.

The two comment corrections clarify the returned logits and configurable city-head output size. Python AST comparison verifies that the edits do not change program structure.

## Deferred work

These items require a separate experiment or behavior-changing maintenance stage:

1. Recover the original run configurations, logs, seeds, checkpoints, and dataset manifests.
2. Verify sample overlap, label mappings, and checkpoint-selection rules.
3. Pin a tested environment and rerun the necessary baseline/evaluation.
4. Add genuine loss/head ablations rather than inferring them from YAML names.
5. Measure TF-SepDualNet parameters/MACs separately from the team's student model.
6. Review the log analysis script: lexical version-directory ordering and clipping negative domain-gap plot limits can affect displayed summaries. They are deliberately not changed here because historical outputs could change.

No statement in this document replaces the poster's historical numbers; it describes how much has been verified in this maintenance pass.
