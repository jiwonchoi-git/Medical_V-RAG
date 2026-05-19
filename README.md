# 🏥 Medical V-RAG: Visual Retrieval-Augmented Generation for Clinical Report Hallucination Reduction

> Reproducing and extending the V-RAG framework from ["Reducing Hallucinations of Medical Multimodal Large Language Models with Visual Retrieval-Augmented Generation"](https://arxiv.org/abs/2502.15040) (Chu et al., 2025, NEC Laboratories America)

---

## Overview

Medical Vision-Language Models (VLMs) are prone to **hallucination**, generating clinically inaccurate content that can affect diagnosis and patient outcomes. Standard text-only RAG partially mitigates this, but assumes retrieved images are perfectly interchangeable with the query image, which is often not the case in radiology.

This project implements **Visual RAG (V-RAG)**, which retrieves both images and their associated reports, allowing the model to determine what is truly relevant from multimodal context.

This work is part of a broader capstone project, **"Toward Trustworthy Clinical AI: A Hallucination-Aware Evaluation Framework for Multimodal Report Generation
"** conducted in collaboration with D4CG at the University of Chicago.

---

## Key Idea

```
Standard RAG:  Query Image → retrieve similar images → use their TEXT only
V-RAG:         Query Image → retrieve similar images → use TEXT + IMAGE together
```

By incorporating both modalities, V-RAG lets the model compare visual features directly, rather than relying solely on textual descriptions of similar cases.

---

## Pipeline

```
[Chest X-ray Images + Radiology Reports]
            ↓
[BiomedCLIP Image Embedding]
            ↓
[FAISS Vector Index (Approximate kNN via HNSW)]
            ↓
[Query Image → Top-k Retrieval (images + reports)]
            ↓
[Multimodal Prompt Construction]
            ↓
[Entity Probing: "Does the patient have [disease entity]?" → Yes/No]
            ↓
[Hallucination Evaluation via F1 Score]
```

---

## Dataset

- **Indiana University Chest X-ray Collection** (OpenI, NIH) — publicly available chest X-ray images and radiology reports
- MIMIC-CXR support planned 

---

## Tech Stack

| Component | Tool |
|---|---|
| Image Embedding | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |
| Vector Database | FAISS (HNSW approximate kNN) |
| LLM | OpenAI GPT-4o |
| NER (Entity Extraction) | Stanza i2b2 biomedical model |
| Pipeline | LangChain |
| Compute | Google Colab (GPU) |

---

## Project Structure

```
medical-vrag/
├── README.md
├── requirements.txt
├── data/
│   └── .gitkeep
├── notebooks/
│   ├── 01_data_prep.ipynb        # Parse Indiana CXR reports + images
│   ├── 02_embedding.ipynb        # BiomedCLIP image embedding + FAISS index
│   └── 03_vrag_pipeline.ipynb   # V-RAG retrieval + entity probing + evaluation
└── src/
    ├── retriever.py              # FAISS-based multimodal retriever
    ├── entity_probe.py           # Disease entity extraction + Yes/No probing
    └── pipeline.py               # End-to-end V-RAG pipeline
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_data_prep` | Parse XML reports, extract findings/impression, match image IDs |
| `02_embedding` | Embed images with BiomedCLIP, build FAISS index |
| `03_vrag_pipeline` | Run V-RAG retrieval, entity probing, compare vs text-only RAG |

---

## Evaluation

Following the original paper, we evaluate using **Entity Probing**:

1. Extract disease entities from reports using biomedical NER
2. For each entity, ask the VLM: *"Does the patient have [entity]?"*
3. Compare Yes/No predictions against LLM-grounded ground truth
4. Report **Precision, Recall, F1**

This provides a clinically meaningful evaluation beyond surface-level metrics like ROUGE.

---

## Related Work

- [V-RAG Paper (Chu et al., 2025)](https://arxiv.org/abs/2502.15040)
- [BiomedCLIP (Zhang et al., 2023)](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- [FAISS (Douze et al., 2024)](https://github.com/facebookresearch/faiss)
- [Indiana University CXR Collection](https://openi.nlm.nih.gov/)
