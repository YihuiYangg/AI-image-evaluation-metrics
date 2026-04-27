import os
import re
import itertools
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import lpips

from transformers import CLIPProcessor, CLIPModel
from pytorch_fid.fid_score import calculate_fid_given_paths


# =========================================================
# 1. CONFIG
# =========================================================

DATASET_ROOT = Path("dataset")
MODEL_DIRS = {
    "gpt": DATASET_ROOT / "gpt",
    "jimeng": DATASET_ROOT / "jimeng",
}

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LPIPS_NET = "alex"  # alex is commonly used and relatively light
FID_BATCH_SIZE = 32
FID_DIMS = 2048

# =========================================================
# 2. PROMPT DEFINITIONS
# =========================================================

PROMPTS = {
    "p01": "A red apple on a wooden table",
    "p02": "A golden retriever sitting in a park",
    "p03": "A blue sports car on a highway",
    "p04": "A white ceramic teacup on a marble surface",
    "p05": "A mountain landscape at sunset",
    "p06": "A busy street market in Tokyo at night",
    "p07": "A futuristic city skyline with flying cars",
    "p08": "A medieval castle surrounded by fog",
    "p09": "An underwater coral reef with colorful fish",
    "p10": "A classroom with students taking an exam",
    "p11": "A cyberpunk street with neon lights",
    "p12": "A rainforest waterfall with mist",
    "p13": "A portrait in impressionist painting style",
    "p14": "A surreal dreamscape with floating islands",
    "p15": "A minimalist black and white abstract artwork",
}

VARIANT_SUFFIX = {
    "A": "top-down overhead view, centered composition",
    "B": "eye-level front view, medium shot",
    "C": "low-angle view from below, wide shot",
    "D": "extreme close-up, shallow depth of field",
}


def full_prompt(prompt_id: str, variant: str) -> str:
    base = PROMPTS[prompt_id]
    suffix = VARIANT_SUFFIX[variant]
    return f"{base}, {suffix}"


# =========================================================
# 3. FILE PARSING
# =========================================================

FILENAME_PATTERN = re.compile(r"^(p\d{2})_([A-D])\.(png|jpg|jpeg|webp)$", re.IGNORECASE)


def list_images(model_name: str, folder: Path):
    rows = []
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    for file in sorted(folder.iterdir()):
        if not file.is_file():
            continue
        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        m = FILENAME_PATTERN.match(file.name)
        if not m:
            print(f"[WARN] Skipping unmatched filename: {file.name}")
            continue

        prompt_id, variant, ext = m.groups()
        prompt_id = prompt_id.lower()
        variant = variant.upper()

        if prompt_id not in PROMPTS:
            print(f"[WARN] Unknown prompt id in filename: {file.name}")
            continue

        rows.append({
            "model": model_name,
            "prompt_id": prompt_id,
            "variant": variant,
            "filename": file.name,
            "filepath": str(file.resolve()),
            "prompt_text": full_prompt(prompt_id, variant),
        })

    return pd.DataFrame(rows)


def build_master_dataframe():
    dfs = []
    for model_name, folder in MODEL_DIRS.items():
        df = list_images(model_name, folder)
        dfs.append(df)

    master = pd.concat(dfs, ignore_index=True)

    if master.empty:
        raise ValueError("No valid images found. Please check folder structure and filenames.")

    return master


# =========================================================
# 4. CLIP SCORE
# =========================================================

def compute_clip_scores(df: pd.DataFrame):
    print("\n[INFO] Loading CLIP model...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    model.eval()

    scores = []

    print("[INFO] Computing CLIP scores...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image = Image.open(row["filepath"]).convert("RGB")
        text = row["prompt_text"]

        inputs = processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            cosine_sim = torch.sum(image_embeds * text_embeds, dim=-1).item()

        scores.append(cosine_sim)

    result = df.copy()
    result["clip_score"] = scores
    return result


# =========================================================
# 5. LPIPS
# =========================================================

def load_image_for_lpips(path: str, device: str):
    image = Image.open(path).convert("RGB").resize((256, 256))
    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
    tensor = tensor.view(image.size[1], image.size[0], 3).permute(2, 0, 1) / 255.0
    tensor = tensor * 2 - 1  # LPIPS expects [-1, 1]
    return tensor.unsqueeze(0).to(device)


def compute_lpips(df: pd.DataFrame):
    print("\n[INFO] Loading LPIPS model...")
    loss_fn = lpips.LPIPS(net=LPIPS_NET).to(DEVICE)
    loss_fn.eval()

    lpips_rows = []

    print("[INFO] Computing LPIPS pairwise scores...")
    for model_name in df["model"].unique():
        df_model = df[df["model"] == model_name]

        for prompt_id in sorted(df_model["prompt_id"].unique()):
            group = df_model[df_model["prompt_id"] == prompt_id].sort_values("variant")
            records = group.to_dict("records")

            if len(records) < 2:
                print(f"[WARN] Not enough images for LPIPS: {model_name} {prompt_id}")
                continue

            for r1, r2 in itertools.combinations(records, 2):
                img1 = load_image_for_lpips(r1["filepath"], DEVICE)
                img2 = load_image_for_lpips(r2["filepath"], DEVICE)

                with torch.no_grad():
                    score = loss_fn(img1, img2).item()

                lpips_rows.append({
                    "model": model_name,
                    "prompt_id": prompt_id,
                    "image_1": r1["filename"],
                    "image_2": r2["filename"],
                    "variant_1": r1["variant"],
                    "variant_2": r2["variant"],
                    "lpips_score": score,
                })

    return pd.DataFrame(lpips_rows)


# =========================================================
# 6. FID
# =========================================================

def compute_fid():
    print("\n[INFO] Computing FID...")
    paths = [str(MODEL_DIRS["gpt"]), str(MODEL_DIRS["jimeng"])]
    fid_value = calculate_fid_given_paths(
        paths=paths,
        batch_size=FID_BATCH_SIZE,
        device=DEVICE,
        dims=FID_DIMS,
    )
    return fid_value


# =========================================================
# 7. SAVE SUMMARIES
# =========================================================

def save_summaries(clip_df: pd.DataFrame, lpips_df: pd.DataFrame, fid_value: float):
    # Save raw outputs
    clip_df.to_csv(OUTPUT_DIR / "clip_scores_per_image.csv", index=False)
    lpips_df.to_csv(OUTPUT_DIR / "lpips_pairwise_scores.csv", index=False)

    # CLIP summary by model
    clip_model_summary = (
        clip_df.groupby("model", as_index=False)["clip_score"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    clip_model_summary.to_csv(OUTPUT_DIR / "clip_summary_by_model.csv", index=False)

    # CLIP summary by prompt and model
    clip_prompt_summary = (
        clip_df.groupby(["model", "prompt_id"], as_index=False)["clip_score"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    clip_prompt_summary.to_csv(OUTPUT_DIR / "clip_summary_by_prompt.csv", index=False)

    # LPIPS summary by model
    lpips_model_summary = (
        lpips_df.groupby("model", as_index=False)["lpips_score"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    lpips_model_summary.to_csv(OUTPUT_DIR / "lpips_summary_by_model.csv", index=False)

    # LPIPS summary by prompt and model
    lpips_prompt_summary = (
        lpips_df.groupby(["model", "prompt_id"], as_index=False)["lpips_score"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    lpips_prompt_summary.to_csv(OUTPUT_DIR / "lpips_summary_by_prompt.csv", index=False)

    # FID summary
    fid_df = pd.DataFrame([{
        "comparison": "gpt_vs_jimeng",
        "fid_score": fid_value
    }])
    fid_df.to_csv(OUTPUT_DIR / "fid_score.csv", index=False)

    print("\n[INFO] Results saved to:", OUTPUT_DIR.resolve())


# =========================================================
# 8. MAIN
# =========================================================

def main():
    print("[INFO] Building master dataframe...")
    master_df = build_master_dataframe()
    master_df.to_csv(OUTPUT_DIR / "image_manifest.csv", index=False)

    print(f"[INFO] Found {len(master_df)} valid images.")
    print(master_df.head())

    clip_df = compute_clip_scores(master_df)
    lpips_df = compute_lpips(master_df)
    fid_value = compute_fid()

    save_summaries(clip_df, lpips_df, fid_value)

    print("\n========== FINAL RESULTS ==========")
    print("\nCLIP mean by model:")
    print(clip_df.groupby("model")["clip_score"].mean())

    print("\nLPIPS mean by model:")
    print(lpips_df.groupby("model")["lpips_score"].mean())

    print(f"\nFID (gpt vs jimeng): {fid_value:.4f}")
    print("===================================\n")


if __name__ == "__main__":
    main()