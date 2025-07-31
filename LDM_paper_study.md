# 📘 Latent Diffusion Models (LDM) 논문 요약: Section 1–4 정리

> High-Resolution Image Synthesis with Latent Diffusion Models  
> Robin Rombach et al., 2022  
> [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

---

## 1. Introduction

### 문제의식
- 기존 DDPM은 매우 강력한 생성 모델이지만, **고해상도 이미지**에 적용하기엔 계산량과 메모리 사용량이 너무 큼.
- 해결책: **고차원 픽셀 공간이 아닌, 압축된 latent space에서 diffusion을 수행**하는 방법 제안

### 핵심 아이디어
- VAE로 이미지를 latent space로 인코딩
- 이 latent 공간에서 DDPM-style diffusion 수행
- 최종적으로 decoder를 통해 이미지를 복원

### 장점
- 계산량/메모리 사용량 감소 (16배 이상 효율적)
- 여전히 고품질 이미지 생성 가능
- 다양한 조건 생성 (텍스트, segmentation map 등)에 확장 용이

---

## 2. Related Work

### 연결된 연구들
- **DDPM, ADM**: pixel space에서 직접 diffusion 수행 → 고비용
- **VAE**: latent 표현 학습에 강점, 하지만 샘플 품질이 낮음
- **Score-based models (NCSN 등)**: 확률적 manifold 학습

### LDM의 위치
- 기존 DDPM의 품질을 유지하면서, VAE의 효율성을 결합
- DDPM과 VAE를 하나의 pipeline으로 연결

---

## 3. Method

### 구조 개요
1. 이미지 → VAE encoder → latent vector \( z \)
2. latent \( z \)에 noise 추가: forward diffusion
3. noised latent \( z_t \)에서 원래 \( z_0 \)를 복원하는 reverse model 학습
4. 최종적으로 decoder로 이미지 복원

---

### Diffusion in Latent Space
- 기존처럼 latent \( z \)에 대해 forward process 정의
- reverse process는 UNet 기반의 조건 생성 네트워크 사용
- noise 예측 방식은 **ε-prediction (DDPM과 유사)**

---

### 손실 함수 구성

LDM은 다양한 loss를 **적절히 혼합**하여 학습 안정성과 품질을 동시에 추구함:

1. **L2 Loss (Reconstruction Loss)**  
   - latent 복원 또는 noise 예측에 사용  
   - 기본적인 학습 안정성을 제공

2. **KL Divergence Loss (VAE 구조)**  
   - encoder의 분포가 \( \mathcal{N}(0, I) \)에 가까워지도록 정규화  
   - latent space의 **샘플링 가능성 보장**

3. **Perceptual Loss**  
   - 이미지 복원 품질을 높이기 위해 사용  
   - VGG feature space 상에서 복원 이미지와 ground truth 이미지의 차이 측정  
   - 인간 시각에 가까운 유사도 추구

4. **Optional Adversarial Loss**  
   - 일부 실험에서 GAN-style discriminator로 fine-tuning 시 사용

---

### Conditioning 구조 (Section 3.3)

- 다양한 조건 입력(텍스트, 세그멘테이션, depth map 등)을 지원
- 주요 방식: **Cross-Attention**

#### Cross-Attention 개념
- Query: 이미지 latent feature
- Key/Value: 조건 정보 (ex. 텍스트 임베딩)
- UNet 내의 Residual Block들에 **반복 삽입**

#### 동작 위치
- **Downsampling, Bottleneck, Upsampling** 모든 stage에서 사용
- 논문에서 명시적으로 설명됨: "*in each residual block at every level of resolution*"

#### 추가 개념: Self-Attention
- latent feature map 내 위치 간 관계 파악
- Cross-Attention과 함께 사용되어 의미적 정합성(semantic alignment) 강화

---

### 🎚️ Classifier-Free Guidance

- 조건 없는 예측과 조건 있는 예측의 차이를 조절
- 수식:

  ```
  epsilon_guided = epsilon_uncond + s * (epsilon_cond - epsilon_uncond)
  ```

- guidance scale \( s \): 조건에 따라 샘플 정합성을 조절하는 하이퍼파라미터
  - \( s \uparrow \): 텍스트 의미 강하게 반영 (정합성↑, 다양성↓)

---

## 4. Experiments

### 주요 목적
- LDM이 DDPM 대비 품질을 유지하면서도 훨씬 효율적인지를 검증
- 다양한 조건 생성(task)에서 잘 작동하는지 평가

---

### 주요 실험

#### 1. **Unconditional Generation**
- Dataset: LSUN, CelebA-HQ, FFHQ 등
- FID 기준 기존 모델과 유사한 성능, 계산량은 크게 감소

#### 2. **Text-to-Image Generation (COCO)**
- 조건: CLIP 텍스트 임베딩
- 결과:
  - FID 4.98 (256×256 기준)
  - 사람 평가에서도 정합성 우수

#### 3. **Super-Resolution**
- 저해상도 이미지 → 고해상도 생성
- Perceptual loss로 학습 → LPIPS, PSNR 개선

#### 4. **Inpainting**
- 이미지 일부 영역 마스킹 → 조건 생성으로 복원
- 매우 자연스러운 텍스처 및 경계 복원 확인

---

### 효율성 비교

| 항목 | LDM | 기존 DDPM |
|------|-----|------------|
| GPU 메모리 | ↓ 12배 | 높음 |
| 학습 속도 | ↑ 16배 빠름 | 느림 |
| 샘플 품질 | 동일 또는 개선 | 우수 |
| 샘플 해상도 | 최대 1024×1024까지 안정적 처리 | 어려움 |

---

## 종합 정리

- LDM은 DDPM의 장점(샘플 품질)을 유지하면서, VAE 기반의 latent space로 이동해 효율성을 극대화한 모델
- cross-attention 구조를 통해 조건 생성에서도 강력함
- 다양한 손실 함수(L2, KL, perceptual loss)를 혼합하여 학습 안정성과 perceptual 품질 모두 확보
- 실험적으로도 텍스트, 이미지, depth 등 다양한 조건에서 SOTA 성능 달성
