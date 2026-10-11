# Work log: SMS spam classification

- 2026-10-11: Loaded the UCI SMS Spam Collection (5,572 messages, 13.4% spam). Found 403 exact duplicate texts with 0 conflicting labels; removed duplicates before splitting (5,169 messages, 12.6% spam).
- 2026-10-11: Stratified 80/20 split, seed 42; stratified 5-fold CV on the training split. Test set sealed.
- 2026-10-11: Baselines: dummy average precision 0.126 (equals the spam share), precision and recall 0 (never predicts spam). Default Naive Bayes (alpha 1.0): average precision 0.942, precision 1.000, recall 0.605. Predicted precision > recall before running; observed.
- 2026-10-11: Tuned MultinomialNB smoothing on training CV only (alpha 1.0, 0.3, 0.1, 0.03, 0.01). Lowering alpha from 1.0 to 0.03 increased average precision from 0.942 to 0.972, recall from 0.605 to 0.885, and F1 from 0.754 to 0.930, with precision remaining above 0.98. Alpha was examined on the same folds it was scored on.
- 2026-10-11: Evaluated probability thresholds at alpha 1.0 using out-of-fold predictions. Threshold 0.10 achieved precision 0.888 and recall 0.883; threshold 0.50 achieved 1.000 and 0.605. Final operating point to be chosen based on acceptable false-positive cost before the sealed-test evaluation.
- 2026-10-11: Working assumption: false positives cost more than false negatives; precision floor of 95% (assumption, not a measured requirement).
