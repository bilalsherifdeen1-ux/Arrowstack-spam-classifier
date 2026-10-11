# SMS Spam Classification (TF-IDF + Multinomial Naive Bayes)

Arrowstack Technologies virtual internship, Machine Learning track (Batch 2026), optional bonus task 3.
Author: Sherifdeen Bilal Olamilekan

> Status: in progress. The sealed test set has NOT been evaluated yet; every number below is cross-validation evidence on the training split. All results are observations on a public dataset, not production claims.

## Problem and stakeholder
The goal is to classify SMS messages as spam or ham (legitimate) so that a messaging filter could block spam while rarely blocking real messages. The working assumption is that a false positive (a legitimate message classified as spam and possibly never seen) costs more than a false negative (a spam message that still reaches the user), so a precision floor of 95% on messages flagged as spam is used as a starting constraint. This is a stated assumption, not a validated business requirement.

This project demonstrates text classification with a simple, inspectable model. It is not a production spam filter.

## Data
- Source: UCI Machine Learning Repository, "SMS Spam Collection" (dataset id 228), downloaded from archive.ics.uci.edu
- Accessed: 2026-10-11
- License: not yet verified; see the UCI dataset page before reuse.
- Raw size: 5,572 messages, 13.4% spam.
- Duplicates: 403 exact duplicate message texts were found, none with conflicting labels. They were removed before splitting, leaving 5,169 messages (12.6% spam). Duplicates were skewed toward spam, and leaving them in would have let identical messages appear in both train and test.
- Split: stratified 80/20, seed 42 (4,135 train / 1,034 test). The test set is sealed.
- No raw data is stored in this repo; the notebook downloads it.

## Method
- Pipeline: TF-IDF (scikit-learn defaults: lowercase, unigrams) followed by Multinomial Naive Bayes. TF-IDF is fit inside the pipeline on training folds only.
- Validation: stratified 5-fold CV on the training split (seed 42).
- Metrics: spam-class precision, recall and F1, plus average precision (PR-AUC). Accuracy is not used because predicting "ham" for everything already scores about 87%.
- Baselines: a dummy classifier (always predicts the majority class) and Naive Bayes with default smoothing (alpha 1.0).

## How to reproduce
1. Open the notebook in Google Colab (or Python 3.13 with the packages in `requirements.txt`; internet access is needed to download the data).
2. Runtime > Run all.
3. Check values (CV, training split): dummy average precision 0.126; Naive Bayes alpha 1.0: average precision 0.942, precision 1.000, recall 0.605; alpha 0.03: 0.972 / 0.981 / 0.885.

## Results so far (cross-validation, training split)

### Baselines and smoothing sweep
| Model | Avg precision | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Dummy (majority class) | 0.126 | 0.000* | 0.000 | n/a |
| Naive Bayes, alpha 1.0 (default) | 0.942 | 1.000 | 0.605 | 0.754 |
| Naive Bayes, alpha 0.3 | 0.966 | 0.995 | 0.820 | 0.899 |
| Naive Bayes, alpha 0.1 | 0.972 | 0.985 | 0.877 | 0.928 |
| Naive Bayes, alpha 0.03 | 0.972 | 0.981 | 0.885 | 0.930 |
| Naive Bayes, alpha 0.01 | 0.969 | 0.973 | 0.897 | 0.933 |

*The dummy never predicts spam, so precision is undefined and reported as 0 (an expected scikit-learn warning).

### Decision threshold at alpha 1.0 (out-of-fold predictions)
| Threshold | Precision | Recall |
|---|---:|---:|
| 0.50 | 1.000 | 0.605 |
| 0.30 | 0.995 | 0.716 |
| 0.20 | 0.986 | 0.797 |
| 0.10 | 0.888 | 0.883 |
| 0.05 | 0.686 | 0.925 |

### Observations
- Default Naive Bayes ranks messages well (average precision 0.942) but misses about 40% of spam at the default threshold (recall 0.605). This is consistent with the large ham prior and default smoothing pushing borderline spam toward ham; this explanation was not directly tested.
- Lowering the smoothing parameter raised recall from 0.605 to 0.885 (alpha 0.03) while precision stayed above 0.98.
- At similar precision (about 0.985), alpha 0.1 reached recall 0.877, versus 0.797 for raising the threshold to 0.2 at alpha 1.0. These are different configurations compared on the same CV folds, so treat the comparison as directional.
- Alpha and threshold were examined on the same CV folds, so some selection optimism is possible.

## Limitations
- The final operating point (alpha and threshold) has not been fixed, and the test set has not been evaluated.
- Only exact duplicates were removed. Near-duplicate messages (for example, differing by a number) can still leak between train and test.
- Single random split; no repeated splits.
- Only the smoothing parameter and the decision threshold were explored; TF-IDF settings (n-grams, stop words) were left at defaults.
- The age and origin of the message collection were not verified, so it may not represent current spam.
- The precision floor of 95% is an assumption, and the cost of false positives was not measured.

## Not for use
- Blocking real users' messages in production without review.
- Languages or message types other than those in the collection.
- Any claim about real-world spam-filter performance.

## Files
- `README.md` this file
- `work_log.md` dated decision log
- `requirements.txt` pinned packages
- the notebook
