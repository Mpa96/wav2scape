"""
Extract XLSR-53 (wav2vec2-large-xlsr-53) features for each HRT segment.

Joins hrt_segments.csv with wav_manifest.csv to locate each segment's
source WAV, extracts the audio slice in-memory, and runs XLSR-53 feature
extraction through the wav2scape AudioProcessor.

Feature types
-------------
xlsr_hidden    : Mean-pooled transformer hidden states (1024-dim per segment)
                 Saved as CSV:  hrt_features_xlsr_hidden_{phase}.csv
xlsr_quantized : Mean-pooled quantized codevectors (768-dim per segment)
                 Saved as CSV:  hrt_features_xlsr_quantized_{phase}.csv
xlsr_codebook  : Normalised per-file codebook usage histogram (102400-dim).
                 This is the exact representation wav2scape uses for its
                 similarity matrix and PCA scatter plots, computed per file
                 instead of per category.
                 Saved as PT:   hrt_features_xlsr_codebook_{phase}.pt
                          CSV:  hrt_features_xlsr_codebook_meta_{phase}.csv
all            : all three

Usage
-----
python src/extract_features.py \\
    --hrt_csv       bop_results/analysis/hrt_segments.csv \\
    --wav_manifest  /path/to/wav_manifest.csv \\
    --out_dir       bop_results/analysis \\
    --feature_type  all \\
    [--phase        {trial1,trial2,both}] \\
    [--model_name   facebook/wav2vec2-large-xlsr-53] \\
    [--cache_dir    /path/to/model/cache]
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

# Ensure src/ is on the path when called as `python src/extract_features.py`
sys.path.insert(0, str(Path(__file__).parent))
from ml.audio_processor import AudioProcessor

TARGET_SR = 16_000
META_COLS = ["participant_id", "chunk_id", "phase", "start_s", "end_s", "duration_s"]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio_slice(wav_path: str, start_s: float, end_s: float) -> tuple[np.ndarray, int]:
    """Read a time slice from a WAV without loading the full file."""
    with sf.SoundFile(wav_path) as f:
        sr = f.samplerate
        start_frame = int(start_s * sr)
        n_frames = int((end_s - start_s) * sr)
        f.seek(start_frame)
        audio = f.read(n_frames, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_join(hrt_csv: str, wav_manifest: str, phase: str) -> pd.DataFrame:
    hrt = pd.read_csv(hrt_csv)
    manifest = pd.read_csv(wav_manifest)

    if phase != "both":
        hrt = hrt[hrt["phase"] == phase].reset_index(drop=True)

    df = hrt.merge(
        manifest[["participant_id", "chunk_id", "phase", "wav_path", "path_resolved"]],
        on=["participant_id", "chunk_id", "phase"],
        how="left",
    )
    # path_resolved may be stored as string "True"/"False" in CSV
    df["path_resolved"] = df["path_resolved"].fillna(False).astype(str).str.lower() == "true"
    n_unresolved = (~df["path_resolved"]).sum()
    print(f"Segments: {len(df)} total, {n_unresolved} with unresolved WAV paths.")
    return df


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_xlsr(
    df: pd.DataFrame,
    processor: AudioProcessor,
    feature_types: list[str],
) -> dict:
    """
    Process every segment in df and collect per-segment XLSR-53 features.

    Returns a dict that may contain:
        'xlsr_hidden'         : pd.DataFrame  (META_COLS + xlsr_h_0 … xlsr_h_1023)
        'xlsr_quantized'      : pd.DataFrame  (META_COLS + xlsr_q_0 … xlsr_q_767)
        'xlsr_codebook'       : torch.Tensor  shape (N, 102400)
        'xlsr_codebook_meta'  : pd.DataFrame  (META_COLS)
    """
    records: list[tuple[dict, dict]] = []
    failed = 0
    n_total = len(df)

    print(f"\nProcessing {n_total} segments:")
    for i, row in df.iterrows():
        if not row["path_resolved"] or not Path(str(row["wav_path"])).exists():
            failed += 1
            continue
        try:
            audio, sr = load_audio_slice(row["wav_path"], row["start_s"], row["end_s"])
            if len(audio) < 400:  # < 25 ms at 16 kHz
                failed += 1
                continue
            feats = processor.extract_features(audio, sr)
            records.append(({c: row[c] for c in META_COLS}, feats))
        except Exception as e:
            warnings.warn(f"Row {i}: failed — {e}")
            failed += 1

        n_done = len(records) + failed
        if n_done % 100 == 0 and n_done > 0:
            print(f"  {n_done}/{n_total} ({n_done/n_total:.0%}), {failed} failed", flush=True)

    print(f"Extraction done: {len(records)} extracted, {failed} failed.")
    if not records:
        return {}

    meta_dicts, feat_dicts = zip(*records)

    results: dict = {}

    if "xlsr_hidden" in feature_types:
        n = len(feat_dicts[0]["hidden"])
        feat_names = [f"xlsr_h_{j}" for j in range(n)]
        rows = [
            {**m, **dict(zip(feat_names, f["hidden"]))}
            for m, f in zip(meta_dicts, feat_dicts)
        ]
        results["xlsr_hidden"] = pd.DataFrame(rows)[META_COLS + feat_names]

    if "xlsr_quantized" in feature_types:
        n = len(feat_dicts[0]["quantized"])
        feat_names = [f"xlsr_q_{j}" for j in range(n)]
        rows = [
            {**m, **dict(zip(feat_names, f["quantized"]))}
            for m, f in zip(meta_dicts, feat_dicts)
        ]
        results["xlsr_quantized"] = pd.DataFrame(rows)[META_COLS + feat_names]

    if "xlsr_codebook" in feature_types:
        results["xlsr_codebook"] = torch.stack(
            [torch.from_numpy(f["codebook"]) for f in feat_dicts]
        )  # (N, 102400)
        results["xlsr_codebook_meta"] = pd.DataFrame(list(meta_dicts))[META_COLS]

    return results


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save(results: dict, out_dir: Path, phase: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, filename in [
        ("xlsr_hidden",    f"hrt_features_xlsr_hidden_{phase}.csv"),
        ("xlsr_quantized", f"hrt_features_xlsr_quantized_{phase}.csv"),
    ]:
        if key in results:
            p = out_dir / filename
            results[key].to_csv(p, index=False)
            r, c = results[key].shape
            print(f"Saved → {p}  ({r} rows, {c} cols)")

    if "xlsr_codebook" in results:
        pt_path   = out_dir / f"hrt_features_xlsr_codebook_{phase}.pt"
        meta_path = out_dir / f"hrt_features_xlsr_codebook_meta_{phase}.csv"
        torch.save(results["xlsr_codebook"], pt_path)
        results["xlsr_codebook_meta"].to_csv(meta_path, index=False)
        print(f"Saved → {pt_path}   (shape {tuple(results['xlsr_codebook'].shape)})")
        print(f"Saved → {meta_path}  ({len(results['xlsr_codebook_meta'])} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract XLSR-53 features for HRT segments."
    )
    parser.add_argument("--hrt_csv",      default="bop_results/analysis/hrt_segments.csv")
    parser.add_argument("--wav_manifest", default="/home/mpa96/PhD/Data/GRASS_BOP_Participants/wav_manifest.csv")
    parser.add_argument("--out_dir",      default="bop_results/analysis")
    parser.add_argument("--phase",        default="both", choices=["trial1", "trial2", "both"])
    parser.add_argument("--feature_type", default="all",
                        choices=["xlsr_hidden", "xlsr_quantized", "xlsr_codebook", "all"])
    parser.add_argument("--model_name",   default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument("--cache_dir",    default=None)
    args = parser.parse_args()

    feature_types = (
        ["xlsr_hidden", "xlsr_quantized", "xlsr_codebook"]
        if args.feature_type == "all"
        else [args.feature_type]
    )

    print(f"Loading model: {args.model_name} ...")
    processor = AudioProcessor(
        model_name=args.model_name,
        sample_rate=TARGET_SR,
        cache_dir=args.cache_dir,
    )

    df = load_and_join(args.hrt_csv, args.wav_manifest, args.phase)
    results = extract_xlsr(df, processor, feature_types)
    save(results, Path(args.out_dir), args.phase)


if __name__ == "__main__":
    main()
