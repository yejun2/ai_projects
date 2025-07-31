# 📘 Denoising Diffusion Probabilistic Models (DDPM): 논문 구조 기반 핵심 정리

> Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley), NeurIPS 2020  
> [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

---

## 1. 서론 (이 논문은 무엇을 새롭게 제안하는가?)

- **새로운 점**: 기존에 주목받지 못했던 diffusion probabilistic model로 고품질 이미지 생성 가능함을 최초로 입증
- **왜 중요한가**: GAN이나 VAE, Flow 기반 모델에 비해 안정적인 학습과 높은 샘플 품질 달성
- **기존 한계**: 기존 diffusion model은 학습은 쉬웠지만 생성 품질이 낮음
- **이 논문은 어떻게 해결했는가**: ε-prediction과 Langevin 해석을 통해 이론적 구조 및 성능 개선
- **무엇을 성취했는가**: CIFAR10 FID 3.17, CelebA-HQ/LSUN 고해상도 샘플링 성공
- **남은 한계**: likelihood는 낮고 속도는 느림

---

## 2. 배경 (기존 연구의 어떤 틈을 메우는가?)

- **새로운 시각**: Diffusion 모델을 latent variable model로 정식화
- **의의**: Variational inference 적용 가능성 확보
- **기존 연구의 부족한 점**: diffusion 구조의 latent 해석 부족
- **이 논문은 어떻게 해결했는가**: forward/reverse process를 Markov chain으로 구조화

---

## 3. 제안 방법 (기존 방법과 다른 점은 무엇이며, 왜 효과적인가?)

- **새로운 구성**: ε-prediction 방식 도입, Langevin dynamics 구조 해석
- **중요성**: 학습 안정성 및 품질 향상
- **기존 방식의 한계**: reverse mean or x₀ prediction은 품질 낮음
- **해결 방식**: ε 예측이 수식 단순화와 score matching과의 연결 제공
- **남은 한계**: 이론적 정당성은 약하고 주로 실험적 증거에 의존

---

## 4. 실험 (어떤 데이터를 사용했고, 어떤 성과를 냈는가?)

- **사용한 데이터**: CIFAR10, CelebA-HQ, LSUN (256×256)
- **주요 성과**:  
  - CIFAR10: FID 3.17, Inception Score 9.46  
  - LSUN Church: FID 7.89, Bedroom: FID 4.90  
- **Ablation 결과**: ε-prediction + simple loss > 다른 구성  
- **한계**: 샘플링 속도 느림, likelihood 낮음

---

## 5. 관련 연구와 비교 (기존 어떤 모델들과 연결되는가?)

- **새로운 연결**: Score-based 모델과 VAE, autoregressive 모델 연결하는 구조 제시
- **중요성**: 생성 모델 간 이론적 통합 가능성 제시
- **기존의 단절**: 생성 모델 간 수식적 연결 부족
- **해결 방식**: ε-parameterization이 score matching과 variational inference 사이의 연결 제공

---

## 6. 결론 및 논의 (무엇을 남겼고, 앞으로 어떤 발전이 필요한가?)

- **의미**: 구조는 단순하지만 강력한 표현력을 갖춘 생성 모델
- **한계**:  
  - 느린 샘플링 속도  
  - 제한된 표현력 (fixed forward process)  
  - conditional decoding 부재
- **향후 방향**:  
  - Fast sampling (DDIM 등)  
  - Conditional generation  
  - Multimodal 확장
