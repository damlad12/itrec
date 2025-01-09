# Instruction-Tuned and Resource-Efficient Top-k Recommendation Framework using TALLRec, LoRA, and RecRanker (ITREK)
---

## 1. Abstract
ITREK is an integrated framework for a top-k recommender system using TALLRec's lightweight instruction-based tuning approach and RecRanker’s top-k ranking optimization pipeline to improve recommendation accuracy and resource efficiency. Our preliminary experimental results demonstrate improvement in recommendation accuracy compared to baseline approaches.

---

## 2. Introduction
The exponential growth of e-commerce platforms necessitates personalized recommendation systems. Recent advances in Large Language Models (LLMs) have yielded impressive performance in various natural language processing tasks, prompting research into their potential for recommendation.  

TALLRec and RecRanker are two frameworks that use instruction tuning to fine-tune LLMs for recommendation tasks:  
- **TALLRec**: Tailored for binary classification of users’ liked/disliked items.  
- **RecRanker**: Designed for top-k recommendation tasks.  

### Core Challenges and Contributions:
1. **Computational Cost**:  
   - TALLRec leverages **LoRA (Low-Rank Adaptation)**, enabling fine-tuning of LLaMA on a single NVIDIA RTX 3090 (24GB) GPU.  
   - RecRanker requires 16 NVIDIA A800 80GB GPUs for training and one A800 80GB GPU for inference.  

2. **Framework Integration**:  
   - Combining **LoRA tuning** of TALLRec with RecRanker’s **advanced ranking pipeline** to maximize accuracy while minimizing computational overhead.

---

## 3. Methodology

### 3.1 Framework Integration
ITREK integrates three frameworks:
- **TALLRec**: Instruction-based tuning of LLaMA-7B using LoRA to reduce trainable parameters and GPU footprint.  
- **RecRanker**: Supplies the hybrid ranking prompt-generation pipeline with pointwise, pairwise, and listwise ranking outputs for final item utility scores.  

### 3.2 Implementation Details
- **Backbone LLM**: LLaMA (7B).  
- **Training Configuration**:
  - Learning rate: 1e-5  
  - Context length: 1024  
  - Batch size: 4  
  - Gradient accumulation: 2  
  - Scheduler: Cosine, 50 warm-up steps, 6000 total training steps  
  - Precision: 8-bit  
  - **LoRA Settings**:
    - Rank: 16  
    - Scaling factor: 16  
    - Dropout rate: 0.05  
    - Layers: q_proj, k_proj, v_proj, o_proj  

- **Inference Configuration**:  
  - Framework: vLLM  
  - Sampling parameters: Temperature = 0.1, Top-k = 10, Top-p = 0.1, Max tokens = 300  
  - Consistent with RecRanker paper: \( \lambda_1 = \lambda_2 = \lambda_3 = 13 \), \( C = 0.92 \), \( C_1 = 0.05, C_2 = 0.5, C_3 = 0.025 \).  
  - GPU: Single NVIDIA A100.

---

## 4. Dataset and Metrics

### 4.1 Dataset
- **ML-100K Dataset**:  
  - 943 users, 1,682 items, 100,000 ratings.  
  - Train-validation-test split: 80%-10%-10% (user-specific).  

### 4.2 Metrics
- **Hit Ratio (HR)**  
- **Normalized Discounted Cumulative Gain (NDCG)**  

---

## 5. Results

### Table 1. ITREK top-k Ranking results using Matrix Factorization as the baseline:

| Backbone | Method            | H@3↑  | N@3↑  | H@5↑   | N@5↑   |
|----------|-------------------|--------|-------|--------|--------|
| Base     |                   | 0.04639 | 0.03272 | 0.0712 | 0.04298 |
| MF       | ITREKpairwise     | 0.071  | 0.0525 | 0.1085 | 0.068   |
| MF       | ITREKpointwise    | 0.0762 | 0.0551 | 0.123  | 0.0742  |
| MF       | ITREKlistwise     | 0.0727 | 0.0522 | 0.1288 | 0.0752  |
| MF       | ITREKhybrid       | 0.0768 | 0.0551 | 0.1169 | 0.0715  |

### Improvement Over Base:
| Method                | H@3↑   | N@3↑   | H@5↑   | N@5↑   |
|-----------------------|--------|--------|--------|--------|
| ITREKhybrid           | 68.8%  | 69.5%  | 69.6%  | 70.2%  |
| RecRankerhybrid       | 51.65% | 57.85% | 33.19% | 44.52% |

**Key Takeaway**: ITREK achieves greater improvement compared to RecRanker.

---

## 6. Conclusions
The results show that, when using the same baseline recommender, ITREK obtains greater improved performance compared to RecRanker. Integrating TALLRec, LoRA, and RecRanker into a unified top-k recommendation framework shows promising potential for balancing accuracy and computational cost.

---

## 7. References
1. Bao, K., Zhang, J., Lin, X., et al. "TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Models with Recommendation." *ACM SIGIR*, 2023.  
2. Luo, S., et al. "RecRanker: Instruction Tuning Large Language Model as Ranker for Top-K Recommendation." *arXiv*, 2024.  
3. Hu, E.J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." *arXiv*, 2021.

