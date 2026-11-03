# Experiment Results

These values were transcribed from user-provided Kaggle output screenshots. Raw run artifacts and model files are not committed. The screenshots show the notebook comparison after configuration and split compatibility checks; this repository does not independently contain the source metrics for re-evaluation.

## Full-data comparison

| Experiment | Notebook | Validation accuracy | Validation loss |
| --- | --- | ---: | ---: |
| Per-frame hand centering | 02 | 42.5908% | 2.560803 |
| Clip-level body reference | 03 | 46.2752% | 2.400093 |
| Body position + local hand shape | 04 | 47.5437% | 2.437259 |

The body-reference run improved validation accuracy by **3.6844 percentage points**. Both use the same `(32, 126)` input dimensions and CNN architecture. Notebook defaults use seed 42, a signer-disjoint split, batch size 64, Adam, and up to 20 epochs. The table reports the evaluated best checkpoints, not necessarily the final epochs.

## Selected class recall

| Class | Hand-centered recall | Body-reference recall | Validation support |
| --- | ---: | ---: | ---: |
| mouth | 3.1579% | 8.4211% | 95 |
| lips | 11.4583% | 35.4167% | 96 |
| chin | 5.4945% | 5.4945% | 91 |
| nose | 9.6774% | 32.2581% | 93 |
| tongue | 37.2340% | 47.8723% | 94 |

The observed gains support preserving signing-location information for this split. They do not establish statistical significance or robustness across other signer splits. Class recall uses each class's own support; raw confusion counts alone should not be interpreted as error rates.

## Combined-feature result

Notebook 04 improved accuracy over notebook 03 by **1.2685 percentage points**, while validation loss increased from 2.400093 to 2.437259. More top-1 predictions were correct, but cross-entropy did not improve. No formal calibration measurement is available.

| Class | Body-reference recall | Combined-feature recall | Validation support |
| --- | ---: | ---: | ---: |
| mouth | 8.4211% | 6.3158% | 95 |
| lips | 35.4167% | 56.2500% | 96 |
| chin | 5.4945% | 36.2637% | 91 |
| nose | 32.2581% | 49.4624% | 93 |
| tongue | 47.8723% | 17.0213% | 94 |

These values also come from user-provided Kaggle screenshots. The combined model is the highest reported single-run accuracy so far, but it has more parameters and substantial class-level regressions. Keep the body-reference checkpoint for comparison.

## Paired training-seed study (fixed signer split)

Values below were transcribed from the latest user-provided Kaggle output screenshots. The signer split is fixed at seed 42; training seeds are 42, 43, and 44. Standard deviations are sample standard deviations across these training runs.

| Representation | Mean validation accuracy | Accuracy standard deviation | Mean validation loss |
| --- | ---: | ---: | ---: |
| Body reference | 46.1452% | 0.3822 percentage points | 2.439975 |
| Body position + local shape | 47.0253% | 0.4031 percentage points | 2.438889 |

| Training seed | Body-reference accuracy | Combined accuracy | Paired gain (percentage points) |
| --- | ---: | ---: | ---: |
| 42 | 46.0466% | 46.5621% | +0.515464 |
| 43 | 46.4635% | 47.2165% | +0.753027 |
| 44 | 45.9256% | 47.2972% | +1.371583 |

Reported paired mean gain: **+0.88 percentage points**, paired gain sample standard deviation: **0.44 percentage points**, combined-model wins: **3/3**. Minor differences from recomputing the table are due to screenshot rounding.

| Class | Mean body-reference recall | Mean combined recall | Support per run |
| --- | ---: | ---: | ---: |
| mouth | 8.4211% | 3.8596% | 95 |
| lips | 52.7778% | 40.2778% | 96 |
| chin | 21.9780% | 18.6813% | 91 |
| nose | 45.8781% | 40.5018% | 93 |
| tongue | 13.4752% | 18.4397% | 94 |

The earlier single-run tongue regression did not persist in the three-seed mean. The selected face-related classes mostly did not account for the overall accuracy gain. Class-level variability remains large, especially for chin.

The seed-42 scores in this study differ from the earlier standalone notebook runs. The screenshots alone do not establish the cause; environment differences, execution state, and nondeterministic operations should be checked against saved artifacts before claiming exact reproducibility. Keep the standalone runs and paired study separate rather than treating them as identical reruns.

These results support a consistent small gain under training randomness on this fixed split. They do not establish significance, robustness across signer cohorts, or a representation-only effect because parameter counts still differ.

## New signer-split study (training seed fixed at 42)

Values were transcribed from user-provided notebook 06 output screenshots. Split seeds 43 and 44 each hold out five signers, with one held-out signer shared between cohorts.

| Split seed | Body-reference accuracy | Combined accuracy | Combined minus body (percentage points) |
| --- | ---: | ---: | ---: |
| 43 | 45.1222% | 44.6977% | -0.424543 |
| 44 | 42.7116% | 42.7028% | -0.008747 |
| Descriptive mean | 43.9169% | 43.7003% | approximately -0.22 |

Mean validation loss was 2.732212 for body reference and 2.759146 for combined features. The combined model won 0/2 paired comparisons; split 44 was effectively tied. Because validation cohorts overlap and only one training seed is used, these are descriptive exploratory results, not independent significance tests.

The fixed-split training-seed gain did not extend to these new signer cohorts. Prefer the simpler body-reference representation (notebook 03) as the working baseline while retaining combined features as an experimental variant. This does not establish that local shape can never help; current evidence does not support its added complexity as a robust improvement.

Selected class recall varied substantially across splits. For example, body-reference mouth recall was 0.325843 on split 43 and 0 on split 44. The screenshots alone do not identify whether the cause is tracking quality, sign variation, or the representation/model.

## Next step: inspect errors before more training

Notebook 07 reads saved validation predictions and raw landmark parquet files without importing TensorFlow or retraining. It verifies sample alignment, selects correct/error examples for mouth, tongue, and awake, summarizes missing landmarks, and renders paired landmark animations. Observations are recorded in a manual review template. These selected examples are development diagnostics, not an unbiased final test or proof of mislabeled data.

## Review of eight raw landmark sequences

Eight user-supplied parquet sequences were inspected locally after notebook 07 diagnostics. Their 543-landmark frame layout and landmark indices matched the expected competition order; source frame identifiers were consecutive within each file. Original video completeness and sign-label correctness were not established.

One mouth error had hand landmarks in only 2 of 6 source frames, and one tongue error in 4 of 6. The existing preprocessing repeats these observations to reach 32 frames, without recovering missing movement. Another tongue example had an internal tracking gap; deleting no-hand rows removes its original time separation. Across these selected clips, the hand-validity mask was constant after no-hand rows were dropped.

One awake-to-wake error had a hand present in all 21 frames, so missingness alone cannot explain every error. No wake reference clips or trained checkpoints were provided for this local review. The review motivates a controlled timing experiment, not a claim that preserving gaps will improve accuracy. Raw landmarks and local review figures are not committed.

Notebook 08 compares drop-grid and full-grid missing-position handling with identical model capacity and normalization. It records visible-row retention and evaluates all validation samples by missingness stratum. Results are pending.
