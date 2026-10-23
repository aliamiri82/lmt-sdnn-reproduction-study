# Validation

Checked on 2026-10-23.

- All included Python sources passed syntax parsing.
- Using the locally installed PyTorch environment, synthetic-input forward passes completed for both teachers and all six student/baseline architectures at both 10-class and 5-class output sizes (16 model configurations).
- A machine-specific teacher-cache path was replaced by a repository-relative `.cache/teachers` default, preserving the LMT_CACHE override.
- Full training, dataset processing, distillation backpropagation, reported accuracy, and historical metrics were not independently rerun. Included metrics/figures are original study artifacts.
- No raw datasets, processed data arrays, checkpoints, or third-party paper PDFs are included.
