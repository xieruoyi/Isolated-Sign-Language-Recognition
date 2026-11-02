"""Small ASL baseline: inspect, prepare, train, and predict.

Run `python asl.py --help`. Paths in config are relative to the config file.
"""
import argparse
import json
from pathlib import Path


def read_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("data_dir", "output_dir"):
        cfg[key] = (path.parent / cfg[key]).resolve()
    return cfg


def metadata(cfg):
    import pandas as pd
    root = cfg["data_dir"]
    if not (root / "train.csv").exists():
        raise FileNotFoundError(
            f"Missing {root / 'train.csv'}. Download/extract the competition data "
            "and set data_dir in config.json."
        )
    table = pd.read_csv(root / "train.csv")
    labels = json.loads((root / "sign_to_prediction_index_map.json").read_text())
    required = {"path", "sign", "participant_id"}
    if not required.issubset(table.columns):
        raise ValueError(f"train.csv must contain {sorted(required)}")
    if sorted(labels.values()) != list(range(len(labels))):
        raise ValueError("Label indices must be contiguous from zero")
    return table, labels


def load_frames(path):
    import numpy as np
    import pandas as pd
    data = pd.read_parquet(path, columns=["x", "y", "z"]).to_numpy(dtype=np.float32)
    if len(data) == 0 or len(data) % 543:
        raise ValueError(f"Invalid landmark rows in {path}: {len(data)}")
    return data.reshape(-1, 543, 3)


def preprocess(frames, length):
    """Official row order: face, left hand, pose, right hand.

    Use xy from both hands. Retain a validity mask; normalize each frame
    relative to the visible hand centroid with a shared xy scale.
    """
    import numpy as np
    hands = frames[:, list(range(468, 489)) + list(range(522, 543)), :2]
    valid = np.isfinite(hands).all(axis=-1)
    useful = valid.any(axis=1)
    hands, valid = hands[useful], valid[useful]
    if not len(hands):
        return np.zeros((length, 126), dtype=np.float32)
    safe = np.where(valid[..., None], hands, 0)
    count = np.maximum(valid.sum(axis=1), 1)[:, None, None]
    center = safe.sum(axis=1, keepdims=True) / count
    centered = np.where(valid[..., None], safe - center, 0)
    scale = np.sqrt((centered ** 2).sum(axis=(1, 2), keepdims=True) / (count * 2))
    normalized = centered / np.maximum(scale, 1e-6)
    features = np.concatenate([normalized.reshape(len(hands), 84), valid], axis=1)
    # Uniform resampling also repeats frames for short clips.
    indices = np.rint(np.linspace(0, len(features) - 1, length)).astype(int)
    return features[indices].astype(np.float32)


def inspect(cfg):
    import numpy as np
    table, labels = metadata(cfg)
    row = table.iloc[0]
    frames = load_frames(cfg["data_dir"] / row["path"])
    print(table.head().to_string(index=False))
    print(f"Samples: {len(table)}, classes: {len(labels)}, signers: {table.participant_id.nunique()}")
    print(f"First sign: {row['sign']}; shape: {frames.shape}; missing: {np.isnan(frames).mean():.2%}")


def prepare(cfg, limit):
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit
    table, labels = metadata(cfg)
    if limit:
        table = table.sample(n=min(limit, len(table)), random_state=cfg["seed"]).reset_index(drop=True)
    if table.participant_id.nunique() < 2:
        raise ValueError("At least two signers are required for validation")
    split = GroupShuffleSplit(n_splits=1, test_size=cfg["validation_fraction"], random_state=cfg["seed"])
    train_idx, val_idx = next(split.split(table, groups=table.participant_id))
    x = np.empty((len(table), cfg["frames"], 126), dtype=np.float32)
    for i, row in enumerate(table.itertuples()):
        x[i] = preprocess(load_frames(cfg["data_dir"] / row.path), cfg["frames"])
        if (i + 1) % 500 == 0:
            print(f"Prepared {i + 1}/{len(table)}", flush=True)
    y = table.sign.map(labels)
    if y.isna().any():
        raise ValueError("Unknown sign in train.csv")
    out = cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "dataset.npz", x=x, y=y.to_numpy(dtype=np.int64), train_idx=train_idx, val_idx=val_idx)
    info = {
        "labels": labels, "frames": cfg["frames"], "features": 126,
        "train_signers": sorted(table.iloc[train_idx].participant_id.astype(str).unique().tolist()),
        "validation_signers": sorted(table.iloc[val_idx].participant_id.astype(str).unique().tolist()),
        "samples": len(table), "smoke_test": bool(limit),
    }
    (out / "metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Saved {len(train_idx)} train / {len(val_idx)} validation samples to {out}")
    print("A limited run checks the pipeline only; use all data for meaningful accuracy.")


def train(cfg):
    import numpy as np
    import tensorflow as tf
    out = cfg["output_dir"]
    info = json.loads((out / "metadata.json").read_text())
    tf.keras.utils.set_random_seed(cfg["seed"])
    with np.load(out / "dataset.npz") as data:
        x, y = data["x"], data["y"]
        tr, va = data["train_idx"], data["val_idx"]
    model = tf.keras.Sequential([
        tf.keras.layers.Input((info["frames"], info["features"])),
        tf.keras.layers.Conv1D(96, 5, padding="same", activation="relu"),
        tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(info["labels"]), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        x[tr], y[tr], validation_data=(x[va], y[va]),
        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(str(out / "best.keras"), monitor="val_accuracy", mode="max", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True),
        ],
    )
    (out / "history.json").write_text(json.dumps(history.history, indent=2))
    scores = model.evaluate(x[va], y[va], return_dict=True)
    (out / "metrics.json").write_text(json.dumps(scores, indent=2))
    print(scores)


def predict(cfg, path):
    import numpy as np
    import tensorflow as tf
    out = cfg["output_dir"]
    info = json.loads((out / "metadata.json").read_text())
    model = tf.keras.models.load_model(out / "best.keras")
    x = preprocess(load_frames(path), info["frames"])
    probs = model.predict(x[None], verbose=0)[0]
    inverse = {v: k for k, v in info["labels"].items()}
    for i in np.argsort(probs)[-5:][::-1]:
        print(f"{inverse[int(i)]:20s} {float(probs[i]):.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="Read metadata and the first landmark sample")
    prep = commands.add_parser("prepare", help="Cache hand features and signer-disjoint split")
    prep.add_argument("--limit", type=int, default=0, help="0 = all samples; 1000 for a smoke test")
    commands.add_parser("train", help="Train a small temporal CNN")
    pred = commands.add_parser("predict", help="Show the top five predictions for a parquet file")
    pred.add_argument("path", type=Path)
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.command == "prepare":
        if args.limit < 0:
            parser.error("--limit must be nonnegative")
        prepare(cfg, args.limit)
    elif args.command == "predict":
        predict(cfg, args.path)
    else:
        globals()[args.command](cfg)


if __name__ == "__main__":
    main()
