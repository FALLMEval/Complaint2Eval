# Complaint2Eval

Complaint2Eval: An Automated Pipeline for Generating LLM Evaluation Criteria from Regulatory Disclosures

## Overview

Large language models (LLMs) are increasingly deployed in high-stakes domains like financial advisory, yet existing evaluation benchmarks rely on synthetic tasks that miss real-world failure modes. **Complaint2Eval** addresses this gap by transforming regulatory complaint data into empirically-grounded evaluation criteria.

### Data

The full dataset consists of over 160,000 FINRA regulatory disclosure records (2000–2020). Due to data use restrictions and the presence of sensitive information, the raw complaint corpus cannot be publicly redistributed. 

### Methodology at a Glance

```
160,000+ FINRA Complaints
         ↓
[Stage 1] Question Discovery 
         ↓ 566 questions
[Stage 2] Semantic Consolidation 
         ↓ 39 criteria
[Stage 3] Rubric Generation (A-D grading scale)
         ↓
[Application]Integrity Score Evaluation
```

### Key Results

| Metric               | Improvement over Baselines     |
| -------------------- | ------------------------------ |
| **Stability (CV)** | ↓ 78% for GPT-4.1 (0.0298→0.0065) |
| **Gender Bias** | ↓ 31-65% (mean \|Bias%\|: 1.72→1.18 vs Vanilla, 3.34→1.18 vs CoT) |
| **Expert Alignment** | Kendall's τ: 0.69-0.96 (8/9 models ≥ 0.78) |
| **Reproducibility** | ICC ≥ 0.99 for 7/9 models |



## Quick Start

**Note**: This project uses [OpenRouter](https://openrouter.ai/) to access multiple LLM providers through a unified API. You can also use direct provider APIs by modifying the configuration in `.env`.

### Generate Evaluation Criteria (Pipeline)

Open and run `Complaint2Eval.ipynb` to:

- Extract questions from regulatory complaints
- Consolidate into evaluation criteria
- Generate graded rubrics

**Output**: `data/criteria/evaluation_criteria.json`

## Repository Structure

- `Complaint2Eval.ipynb`: Complete pipeline implementation

## Citation
```bibtex
@inproceedings{complaint2eval2026,
  title={Complaint2Eval: An Automated Pipeline for Generating LLM Evaluation Criteria from Regulatory Disclosures},
  author={Anonymous},
  year={2026}
}
```

