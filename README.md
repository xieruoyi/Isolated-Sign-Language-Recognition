# Isolated Sign Language Recognition

Recognize isolated American Sign Language (ASL) signs from MediaPipe landmark sequences. This project uses the [Google Isolated Sign Language Recognition competition](https://www.kaggle.com/competitions/asl-signs) dataset, which contains a vocabulary of 250 signs.

## Recommended workflow: VS Code with a Kaggle kernel

Write and edit the notebook in VS Code while running its cells on a Kaggle Jupyter Server. The remote kernel provides compute and access to competition data. Local terminal commands still run on your computer.

1. Clone this repository and open the folder in VS Code.
2. Open your Kaggle notebook and add the `asl-signs` competition dataset through **Add Input**. Complete any required data access steps in your own Kaggle account.
3. Copy the **VSCode Compatible URL** from Kaggle's Jupyter Server panel.
4. Open `notebooks/01_kaggle_explore.ipynb` in VS Code. Choose **Select Kernel**, then **Existing Jupyter Server** (under **Select Another Kernel** if needed), and select the Kaggle Python kernel.
5. Run the notebook cells in order.

The exploration notebook is self-contained. It does not need a local virtual environment, `asl.py`, or `config.json` on the remote server. It discovers the dataset under `/kaggle/input`, including nested competition directories such as `/kaggle/input/competitions/asl-signs`. If more than one matching dataset is found, set `DATA_DIR` explicitly in the notebook.

Keep the server URL private: it may contain session credentials. Do not commit it to Git.

## What the exploration notebook does

- Checks the runtime and discovers the dataset directory.
- Loads `train.csv` and the official label-to-index mapping.
- Displays sample, class, and participant counts.
- Reads a landmark sequence with shape `(frames, 543, 3)`.
- Extracts both hands, handles missing coordinates, and produces 32 frames of 126 features: 84 normalized xy coordinates and 42 validity indicators.
- Plots wrist motion and visible hand landmarks from a selected frame.

The wrist plot is saved to `/kaggle/working/asl-project/first_sample_wrist_motion.png`. Download important outputs or save a Kaggle version with outputs before ending the session. Exploration does not train a model and does not require a GPU.

## Repository files

| File | Purpose |
| --- | --- |
| `notebooks/01_kaggle_explore.ipynb` | Standalone data exploration using a Kaggle remote kernel |
| `notebooks/02_kaggle_train_baseline.ipynb` | Standalone Kaggle CNN training, validation, and experiment artifacts |
| `asl.py` | Command-line baseline for inspection, preprocessing, training, and prediction |
| `config.json` | Data paths and baseline training settings for `asl.py` |
| `requirements.txt` | Dependency ranges for an optional local environment |
| `requirements-lock.txt` | Package versions captured from a Windows Python 3.12 environment |

The lock file is a local environment snapshot, not a specification for Kaggle's preinstalled environment. Dataset files, virtual environments, trained models, and generated artifacts are excluded from Git.

## Train a baseline on Kaggle

Open `notebooks/02_kaggle_train_baseline.ipynb` using the same Kaggle remote kernel. It is self-contained and discovers the attached competition data automatically.

The default configuration runs a 1,000-sample pipeline check for 2 epochs. It splits by signer before sampling, caches features on disk, trains a small temporal CNN, and saves the best checkpoint, validation accuracy, per-class report, predictions, and learning curves in a unique run directory under `/kaggle/working/asl-project`.

After the check succeeds, set `SAMPLE_LIMIT = None` and `EPOCHS = 20` in the notebook and rerun from the configuration cell. Save important artifacts before ending the Kaggle session. Small-subset accuracy is not a benchmark for the full task.

## Optional: run the baseline locally

Use Python 3.11 or 3.12 for the TensorFlow version range in this project. In a Windows PowerShell terminal:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Select `.venv` through **Python: Select Interpreter** if you want to run locally. Keep the Kaggle remote kernel selected for the recommended notebook workflow.

Download and extract the competition data into this layout, or change `data_dir` in `config.json` to an existing dataset directory:

```text
data/asl-signs/
  train.csv
  sign_to_prediction_index_map.json
  train_landmark_files/
    <participant_id>/
      <sequence_id>.parquet
```

Paths in `config.json` are relative to that file unless absolute. Forward slashes work well for Windows paths, for example `D:/datasets/asl-signs`.

Run the following commands from the repository root:

```powershell
# Inspect metadata and one sequence.
.\.venv\Scripts\python.exe asl.py inspect

# Check the pipeline with a small subset first.
.\.venv\Scripts\python.exe asl.py prepare --limit 1000
.\.venv\Scripts\python.exe asl.py train

# Rebuild the cache using all samples, then train again.
.\.venv\Scripts\python.exe asl.py prepare
.\.venv\Scripts\python.exe asl.py train

# Replace the placeholders with an actual sample path.
.\.venv\Scripts\python.exe asl.py predict "data/asl-signs/train_landmark_files/<participant_id>/<sequence_id>.parquet"
```

The baseline saves cached features, the best model, training history, and validation metrics under `artifacts/`. Repeated runs overwrite files with the same names; copy important experiment outputs before rerunning. Full-dataset preprocessing and training require several GB of RAM. Use a limited subset to check the pipeline first.

To run `asl.py` on Kaggle instead, transfer the script and configuration to the remote environment, set `data_dir` to the actual Kaggle input directory and `output_dir` to a writable path under `/kaggle/working`, and invoke the script there. Connecting a notebook to a remote kernel does not automatically transfer sibling project files.

## Evaluation and current scope

The baseline splits data by `participant_id`, keeping each signer in either training or validation. This helps evaluate performance on unseen signers. A 1,000-sample run is only a pipeline check and is not a meaningful benchmark for the full 250-class task.

The exploration notebook, Kaggle training notebook, and baseline script are available. Model training and full competition evaluation have not been verified on the real dataset as part of repository setup.

This repository does not yet generate a competition-ready `submission.zip`. Submission requires a TensorFlow Lite model that accepts raw landmark sequences, includes compatible preprocessing, preserves the official output class order, and meets the competition's runtime, size, signature, and latency requirements. The current baseline accepts preprocessed features and cannot be submitted directly.

Real-time camera recognition and sentence translation are outside the current scope. The next step is to execute the training notebook and record a measured CNN baseline on Kaggle using its signer-disjoint validation split.

## Data attribution

Competition: Ashley Chow, Glenn Cameron, Mark Sherwood, Phil Culliton, Sam Sepah, Sohier Dane, and Thad Starner. *Google - Isolated Sign Language Recognition*. Kaggle, 2023.

The dataset was provided by the Deaf Professional Arts Network and the Georgia Institute of Technology under CC-BY 4.0. Follow the original dataset terms and retain attribution when using or redistributing it. Data is not included in this repository.
