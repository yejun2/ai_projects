# 📘 Denoising Diffusion Probabilistic Models (DDPM): 논문 구조 기반 심화 요약

> Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley), NeurIPS 2020  
> [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

---

## 1. Introduction (이 논문은 무엇을 새롭게 제안하는가?)

- **새로운 제안**: 기존에는 실용성이 떨어진다고 여겨졌던 diffusion probabilistic model이 실제로는 매우 높은 품질의 이미지를 생성할 수 있다는 것을 처음으로 입증함. 특히, ε을 예측하는 방식의 학습 구조를 도입하고, 샘플링 과정을 Langevin dynamics 형태로 해석함.
- **왜 중요한가**: GAN이나 VAE처럼 복잡한 구조 없이도 단순한 노이즈 삽입과 제거만으로 우수한 생성 성능을 낼 수 있음을 보여줌. 학습도 안정적임.
- **기존 한계**: 이전까지 diffusion 모델은 느리고 품질이 낮아 실제 이미지 생성에는 부적합하다고 여겨짐.
- **어떻게 해결했는가**: denoising score matching과의 연결성을 기반으로 학습 구조를 재설계하고, 변형된 목적함수를 사용해 효율성과 성능을 동시에 확보함.
- **무엇을 성취했는가**: CIFAR10에서 FID 3.17, CelebA-HQ 및 LSUN에서도 ProgressiveGAN과 유사한 수준의 고품질 이미지 생성.
- **남은 문제점**: 샘플 품질은 높지만, 로그 가능도는 다른 likelihood 기반 모델보다 낮고, 샘플링 속도는 여전히 느림.

---

## 2. Background (이 논문은 어떤 이론적 틈을 메우는가?)

- **새로운 시각**: diffusion 모델을 latent variable model의 형태로 재정의하여 variational inference의 적용이 가능하게 함.
- **왜 중요한가**: 기존의 score-based 모델과 VAE 같은 확률적 모델을 이론적으로 연결할 수 있는 틀을 제공.
- **기존의 공백**: diffusion 모델은 기존 생성모델들과 수식적으로 연결되지 않았으며, 명확한 확률적 해석이 부족했음.
- **어떻게 해결했는가**: forward/reverse 과정 모두를 Gaussian Markov Chain으로 정의하고, 이 위에 변분 추론을 적용함.

---

## 3. Method (기존 방식과 다른 점은 무엇이며, 왜 효과적인가?)

- **핵심 아이디어**: 데이터를 복원하는 대신, 추가된 노이즈인 ε을 예측하도록 학습 구조를 변경함. 이 단순한 변경이 안정성과 성능을 크게 향상시킴.
- **왜 효과적인가**: 이 방식은 denoising score matching과 수식적으로 유사하여 해석이 쉬우며, 각 시간 단계별로 적절한 손실 비중을 반영할 수 있음.
- **기존 방식의 문제점**: x₀ 자체를 예측하거나 reverse mean을 직접 예측하는 방식은 불안정하거나 품질이 낮았음.
- **해결 방식**: ε 예측은 목적함수를 단순한 평균제곱오차 형태로 바꾸어 학습이 쉬워지고 이론적으로도 설명 가능함.
- **주의점**: ε 예측이 항상 우월하다는 이론적 보장은 부족하고, 데이터나 과제에 따라 성능 차이가 날 수 있음.

---

## 4. Experiments (어떤 데이터를 사용했고, 무엇을 성취했는가?)

- **사용한 데이터셋**: CIFAR10 (32x32), CelebA-HQ (256x256), LSUN Church, Bedroom, Cat (256x256)
- **실험 결과**:
  - CIFAR10에서 FID 3.17, Inception Score 9.46
  - LSUN Church FID 7.89, Bedroom FID 4.90 등
- **Ablation 분석**:
  - ε 예측과 단순화된 손실 함수 조합이 가장 좋은 샘플 품질을 제공
  - reverse variance를 학습시키면 오히려 불안정성 유발
- **한계**:
  - 샘플링 속도는 여전히 느리며 (1000 step)
  - likelihood 기반 모델로서는 정확도가 낮음

---

## 5. Related Work (이 논문은 기존 어떤 모델과 연결되는가?)

- **연결된 모델들**: score-based 모델(NCSN), VAE, autoregressive 모델 등과의 연결 구조를 제시
- **의의**: 다양한 생성 모델들 사이의 수식적 연관성과 전이 가능성을 보여주며, 이론적 통합 기반을 마련함.
- **기존 한계**: 생성 모델 간에 명확한 이론적 연결이 부족했음
- **해결 방법**: ε 예측을 통해 score matching이 variational inference와 수식적으로 동일하다는 점을 보여줌

---

## 6. Conclusion and Discussion (무엇을 남기고, 앞으로 무엇을 개선해야 하는가?)

- **요약**: DDPM은 간단하지만 강력한 구조를 가진 생성 모델로, GAN과 달리 학습이 안정적이고 수학적으로도 해석 가능함.
- **한계점**:
  - 샘플링 속도가 느림
  - 표현력이 제한됨 (fixed decoder, 조건부 생성 없음)
- **향후 과제**:
  - 샘플링 단계 축소 (예: DDIM)
  - 조건부/다중 모달 생성 기능 확장
  - ε 예측 방식의 이론적 정당성 분석

