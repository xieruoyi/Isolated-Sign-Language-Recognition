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

## Next experiment

Notebook 04 combines body-reference coordinates with independently normalized local hand shape. It keeps the split and training settings, but doubles the feature count to 252 and increases the first convolution's parameter count. The single-run results are recorded above. Notebook 05 repeats both variants across three paired training seeds with a fixed signer split, reports optimization variability, and compares tongue predictions on the same samples. Results of that repeated-seed study are pending. A future parameter-matched ablation and new signer splits would be needed to assess capacity effects and broader robustness.
