# Experiment Results

These values were transcribed from user-provided Kaggle output screenshots. Raw run artifacts and model files are not committed. The screenshots show the notebook comparison after configuration and split compatibility checks; this repository does not independently contain the source metrics for re-evaluation.

## Full-data comparison

| Experiment | Notebook | Validation accuracy | Validation loss |
| --- | --- | ---: | ---: |
| Per-frame hand centering | 02 | 42.5908% | 2.560803 |
| Clip-level body reference | 03 | 46.2752% | 2.400093 |

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

## Next experiment

Notebook 04 combines body-reference coordinates with independently normalized local hand shape. It keeps the split and training settings, but doubles the feature count to 252 and increases the first convolution's parameter count. Results are pending. A future parameter-matched ablation or repeated-seed evaluation would be needed to isolate representation and capacity effects more carefully.
