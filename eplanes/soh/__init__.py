"""Physics-based, BMS-independent SOH label and hybrid prediction models.

Stages (python -m eplanes.soh --help):
  extract   per-session observables from the analysis-ready 10 Hz data (resumable)
  label     voltage-anchored capacity, rest-to-rest capacity, resistance -> state-space SOH label
  validate  tests of the label that need no ground truth
  train     hybrid models on the training fleet only (rolling-origin CV, ablation runs)
  test-final  score the held-out fleet once (guarded by a lock file)
  report    figures, summary and a results ZIP
"""
