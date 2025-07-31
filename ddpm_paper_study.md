# 📘 Denoising Diffusion Probabilistic Models (DDPM): 논문 구조 기반 핵심 정리

> Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley), NeurIPS 2020  
> [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

---

## 1. Introduction

- **What’s new**: 기존에 주목받지 못했던 diffusion probabilistic model로 고품질 이미지 생성 가능함을 최초로 입증
- **Why it matters**: GAN이나 VAE, Flow 기반 모델에 비해 안정적인 학습과 높은 샘플 품질 달성
- **Gap**: 기존 diffusion model은 학습은 쉬웠지만 생성 품질이 낮음
- **How filled**: ε-prediction과 Langevin 해석을 통해 이론적 구조 및 성능 개선
- **Achievements**: CIFAR10 FID 3.17, CelebA-HQ/LSUN 고해상도 샘플링 성공
- **Limitations**: likelihood는 낮고 속도는 느림

---

## 2. Background

- **What’s new**: Diffusion 모델을 latent variable model로 정식화
- **Why it matters**: Variational inference 적용 가능성 확보
- **Gap**: 기존에는 diffusion 구조의 latent 해석 부족
- **How filled**: forward/reverse process를 Markov chain으로 구조화

---

## 3. Method

- **What’s new**: ε-prediction 방식 도입, Langevin dynamics 구조 해석
- **Why it matters**: 학습 안정성 및 품질 향상
- **Gap**: 기존 reverse mean or x₀ prediction은 품질 낮음
- **How filled**: ε 예측이 수식 단순화와 score matching과의 연결 제공
- **Limitations**: 이론적 정당성은 약하고 주로 실험적 증거에 의존

---

## 4. Experiments

- **Data used**: CIFAR10, CelebA-HQ, LSUN (256×256)
- **Achievements**:  
  - CIFAR10: FID 3.17, Inception Score 9.46  
  - LSUN Church: FID 7.89, Bedroom: FID 4.90  
- **Ablation**: ε-prediction + simple loss > 다른 구성  
- **Limitations**: 샘플링 속도 느림, likelihood 낮음

---

## 5. Related Work

- **What’s new**: Score-based 모델과 VAE, autoregressive 모델 연결하는 구조 제시
- **Why it matters**: 생성 모델 간 이론적 통합 가능성 제시
- **Gap**: 기존엔 구조 간 연결성 부족
- **How filled**: ε-parameterization이 score matching과 variational inference 사이의 연결 제공

---

## 6. Discussion & Conclusion

- **Why it matters**: 구조 단순하지만 표현력 강한 생성 모델
- **Limitations**:  
  - 느린 샘플링  
  - 제한된 표현력 (fixed forward process)  
  - conditional decoding 미지원
- **Future direction**:  
  - Fast sampling (DDIM 등)  
  - Conditional generation  
  - Multimodal 확장

