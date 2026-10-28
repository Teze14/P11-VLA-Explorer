# VLA Explorer — Project P11

**A working educational toolkit for connecting visual evidence, language, and an explicit 2D action proposal.**

Team: Robert Norris, Jesus Garcia, Michelle Antunez · University of Arizona · ECE 425/525.

## What this project contributes

The contribution is an inspectable learning experience, not training a new foundation model. Real frozen CLIP encoders compare image regions with a user instruction. The UI exposes embeddings' shapes, tokenization, cosine scores, temperature softmax, weighted target coordinates, and the final action rule. Six guided lessons, four experiments, saved comparisons, occlusion, and JSON exports support learning and evaluation.

**Scientific scope:** this is a simplified vision-language-action teaching pipeline. CLIP is a vision-language model; the action head here is deterministic, not trained on robot demonstrations. The heatmap is region–text relevance, not internal attention, segmentation, or a causal explanation. No real robot is controlled. See `docs/SCIENTIFIC_SCOPE.md`.

## Run locally (Windows, macOS, or Linux)

1. Install Python 3.10+ and use a recent desktop Chrome or Edge browser. No Python ML packages, CUDA, API key, or paid service is needed for the browser toolkit.
2. Open a terminal in this folder and run:

```text
python serve.py
```

3. Open **http://127.0.0.1:8767**. Click **Load model & explore**. The first run fetches approximately 154 MB of model weights from Hugging Face if local files are absent. The browser also needs about 22 MB of inference runtime, included in this project. Allow time for download and initialization.
4. For a reliable classroom demonstration, download model files in advance:

```text
python download_models.py
python serve.py
```

Git clones and portable source archives omit model weights; the download script restores the pinned revision. The original demo folder may already contain them. Browser inference runs locally; images and prompts are not sent to a model server. Camera access is an optional snapshot feature, not continuous real-time video.

Do not open `index.html` by double-clicking: ES modules and WebAssembly need HTTP. If a port is occupied, use `python serve.py --port 8768` and visit that port. A model error is displayed explicitly; there is no fabricated-output fallback.

## A 60-second first experiment

1. Keep the synthetic scene and 3 × 3 grid; run **a red cup**.
2. Save a comparison. Change only the text to **a blue cube** and run again.
3. Open the **Similarity** pipeline step to inspect actual scores.
4. Change temperature, then opacity. Only temperature should change the numerical weights.
5. Save and export the run. Try **Occlude strongest region & rerun**, then restore the image.

The weighted target can fall between objects. A target-absent command still receives normalized weights. These are deliberate learning cases, not evidence of reliable robot control.

## Reproducibility

- Runtime: Transformers.js **3.8.1**, bundled locally; ONNX Runtime Web WASM, one thread.
- Model: `Xenova/clip-vit-base-patch32`, revision `d15189d7028b43f1d3e65039190477f6af591c2a`; q8 text and vision projections; base `openai/clip-vit-base-patch32`.
- Each disjoint grid crop is resized to 224 × 224 before CLIP preprocessing. This square stretching is a documented simplification, not the original whole-image preprocessing protocol.
- Both embeddings are L2-normalized. Cosine similarities are divided by user temperature and normalized with stable softmax. This temperature is a teaching control, not CLIP's learned logit scale.
- Action: weighted crop center; delta = gain × (target − [0.5, 0.88]). Coordinates are normalized image coordinates, +x right and +y down.
- Local model manifest records file sizes and SHA-256 hashes. `evaluation/` contains measured timings and functional checks, with measurement conditions.
- `requirements.txt` documents the standard-library-only launcher. Optional notebook dependencies are separate.

## Project contents

| Folder/file | Purpose |
|---|---|
| `toolkit/` | Runnable browser app, model module, lessons, original scene, licensed example photo |
| `serve.py`, `launch_windows.cmd` | Local launch |
| `download_models.py` | Pinned model acquisition for offline/local demos |
| `build_site.py` | Creates a static hosting build without large model files |
| `docs/` | Scientific scope, teacher guide, evaluation plan, roadmap, reading notes |
| `evaluation/` | Actual local inference measurements and QA evidence |
| `notebooks/` | Optional genuine-VLA integration exercise; execution status stated inside |

## Repository

Source code: https://github.com/Teze14/P11-VLA-Explorer (branch `main`).

```text
git clone https://github.com/Teze14/P11-VLA-Explorer.git
cd P11-VLA-Explorer
python download_models.py
python serve.py
```

Model weights are downloaded separately and excluded from Git. The repository is public, so course staff can access the source without an invitation.

## Sources and credits

CLIP: Radford et al. (2021), https://arxiv.org/abs/2103.00020. ONNX export: https://huggingface.co/Xenova/clip-vit-base-patch32. OpenVLA: https://openvla.github.io/. SmolVLA: https://huggingface.co/lerobot/smolvla_base. Teaching design informed by the professor's six sample papers; see `docs/SAMPLE_PAPER_NOTES.md`.

The synthetic scene is original code-generated artwork. Coffee photograph: Rachel Michetti, CC0, via scikit-image; see `toolkit/assets/README.md`. Third-party runtime licenses are retained in `toolkit/vendor/`. Model weights retain their upstream terms.

AI tools assisted implementation, drafting, and verification. Team members must review and understand the deliverables. No student study, robot trial, or educational effectiveness result is claimed.
