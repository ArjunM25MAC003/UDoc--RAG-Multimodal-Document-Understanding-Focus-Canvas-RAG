# UDoc++ Thesis Experiments

This repository contains an end-to-end research notebook for evaluating **UDoc++** on document understanding and long-document question answering. The notebook includes data preparation, model training, retrieval, multimodal inference, evaluation, and thesis-ready exports.

## Research tracks

| Track | Task | Datasets | Main evaluation |
| --- | --- | --- | --- |
| **1 - Region classification** | Classify semantic regions in document images | [FUNSD](https://huggingface.co/datasets/nielsr/funsd-layoutlmv3), [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | Weighted and macro F1, per-class scores, confusion matrices |
| **2 - Long-document QA** | Retrieve evidence and answer questions over multi-page documents | [MMLongBench-Doc](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc), [MP-DocVQA](https://huggingface.co/datasets/lmms-lab/MP-DocVQA) | ANLS, evidence-page recall, visual-page use, latency |

The experiments compare four Track 1 model variants and four Track 2 answer modes: text-only, full-page VLM, cropped-region VLM, and UDoc++-RAG. Track 2 splits are made by document ID to keep questions from the same document on one side of the train/evaluation boundary.

## Notebook

Open [`UDoc_Thesis_End_to_End.ipynb`](UDoc_Thesis_End_to_End.ipynb) in Jupyter and run its cells **in order, in one kernel**. The notebook is designed for a CUDA GPU allocation; Track 2 VLM evaluation requires CUDA.


Use the CUDA-enabled environment prepared for the cluster. At startup, the notebook checks for its required Python packages and configures Hugging Face, PyTorch, EasyOCR, and Matplotlib caches beneath the output workspace. It does not install packages or replace the environment's `torch` or `torchvision` builds.

## Run configuration

The first code cell defines the run settings:

- `FAST_CHECK = True` runs a small pipeline check (including two Track 1 epochs). Treat these results as a functional smoke run, not final thesis results.
- Set `FAST_CHECK = False` for the larger configured run (20 Track 1 epochs, more documents and questions, and longer retriever training).
- `OUTPUT_ROOT` sets the directory for caches, checkpoints, intermediate files, and final exports. If unset, the notebook uses its configured scratch path when available, or `~/omnidoc_outputs`.
- `UDOC_VLM_ID` optionally overrides the default `Qwen/Qwen3-VL-4B-Instruct` model.
- Set `UDOC_VLM_4BIT=1` to request 4-bit VLM loading; this requires `bitsandbytes` in the active environment.

The notebook fixes the random seed to 42 and records sample counts, document split IDs, model IDs, software versions, and settings in its run metadata. Dataset and model downloads, OCR, and training may take substantial time and storage. Record the output metadata alongside any reported results; metrics can vary with dataset revisions and sampled documents.

## Workflow

1. Check the active environment and GPU; configure the output workspace.
2. Load FUNSD and CORD-v2 and run the four region-classification variants for each dataset. Select checkpoints using a validation subset and evaluate the test split once.
3. Prepare document-disjoint samples of MMLongBench-Doc and MP-DocVQA. Convert MP-DocVQA page images to PDFs for the shared document pipeline.
4. Parse PDF text, tables, and figures; use EasyOCR for sparse-text pages; build cross-page document graphs and Focus-Canvas examples.
5. Train the retriever and visual gate, index evaluation documents, and run the Track 2 answer modes.
6. Summarize metrics and save predictions, tables, figures, checkpoints, and run metadata.

## Outputs

All generated files are written beneath `OUTPUT_ROOT` (or the notebook's default workspace). The notebook produces:

- Model checkpoints and cached dataset/model downloads
- Track 1 classification summaries and confusion matrices
- Track 2 per-question predictions and aggregate QA metrics
- Thesis tables, plots, and Focus-Canvas examples
- `run_metadata.json` with run configuration, data counts, split IDs, and software versions

Keep the complete output workspace for each final experiment so results can be traced to the corresponding configuration and evaluation documents.

