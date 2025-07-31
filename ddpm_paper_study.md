Denoising Diffusion Probabilistic Models (DDPM)

Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley), NeurIPS 2020
arXiv:2006.11239

⸻

📌 1. 핵심 아이디어 요약
	•	DDPM은 Markov Chain 기반 latent variable model로, 데이터를 점차적으로 노이즈화한 후 역방향으로 복원하는 방식.
	•	Forward Process: Gaussian noise를 점진적으로 추가.
	•	Reverse Process: 학습된 neural network (U-Net 기반)가 해당 노이즈를 제거해 원본을 복원.
	•	Training Objective는 variational bound를 기반으로 하며, 이 과정에서 denoising score matching과 Langevin dynamics가 연결됨.

📎 DDPM이 latent variable model인 이유

“진짜 latent variable model인가?” 라는 질문에 답하기 위해선 구조와 수식 기반 해석이 필요함.

	•	일반적인 latent variable model은 다음과 같은 구조를 가짐:
$$ p(x) = \int p(x|z)p(z)dz $$
	•	DDPM도 다음과 같이 표현됨:
$$ p_\theta(x_0) = \int p_\theta(x_{0:T}) dx_{1:T} $$
	•	여기서 $x_1, …, x_T$는 data와 동일 차원의 연속 latent로 해석됨 → trajectory 기반 latent structure
	•	Forward는 encoder 역할 (노이즈를 추가), Reverse는 decoder 역할 (복원)
	•	$x_T \sim \mathcal{N}(0, I)$ 는 prior 역할을 수행

🧩 결국 DDPM은:
	•	latent는 하나의 벡터가 아니라 연속적인 노이즈 상태들 $x_{1:T}$
	•	이들을 통해 $x_0$을 생성하며, variational inference를 통한 학습을 수행
	•	따라서 명백한 latent variable model의 구조를 따름

⸻

🧠 2. 구조 개요 (Forward & Reverse Process)

2.1 Forward process (Diffusion)
	•	정의:
$$ q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1 - \beta_t} x_{t-1}, \beta_t I) $$
	•	점차적으로 데이터에 노이즈를 더해 신호를 완전히 파괴
	•	$$ q(x_t | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I) $$

2.2 Reverse process (Sampling)
	•	목표: 노이즈가 있는 $x_t$로부터 $x_{t-1}$을 샘플링해 $x_0$을 복원
	•	$$ p_\theta(x_{t-1} | x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t)) $$

✅ Forward는 고정된 확률 분포, Reverse는 학습되는 neural network로 구성됨

⸻

🔍 3. 학습 목적 함수와 수식 구조

3.1 Variational Lower Bound (VLB)
	•	원래 objective:

$$
\mathcal{L} = \mathbb{E}q\left[ D{KL}(q(x_T|x_0) \Vert p(x_T)) + \sum_{t>1} D_{KL}(q(x_{t-1}|x_t,x_0) \Vert p_\theta(x_{t-1}|x_t)) - \log p_\theta(x_0|x_1) \right]
$$
	•	각 항의 의미:
	•	$L_T$: prior와 noise의 KL → 상수로 무시 가능
	•	$L_{t-1}$: 역방향 조건 분포에 대한 matching
	•	$L_0$: 최종 복원 이미지에 대한 decoding likelihood
	•	$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$
	•	목표: $\epsilon \sim \mathcal{N}(0, I)$ 를 예측하도록 학습
	•	새로운 loss:

$$
\mathcal{L}{\text{simple}} = \mathbb{E}{t, x_0, \epsilon} \left[ | \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) |^2 \right]
$$

🎯 수식적으로는 간단하지만, 이 parameterization이 왜 성능이 좋은지 깊이 이해가 필요함

⸻

🚧 4. 내가 어려움을 겪었던 개념들

4.1 ε-prediction 방식
	•	$\mu_\theta$ 대신 $\epsilon_\theta$을 예측하면:
	•	수식이 단순화되고 학습이 안정적임
	•	denoising score matching과 유사한 형태로 해석 가능
	•	Langevin dynamics와 유사한 샘플링 해석이 가능

내가 썼다면? → 학습의 효율성과 샘플 품질을 위해 수식의 목적함수를 간단히 바꾸는 시도를 했을 것.

4.2 시간 임베딩의 역할
	•	시간 $t$에 따라 노이즈 레벨이 달라지기 때문에, neural network에 반드시 알려줘야 함
	•	sinusoidal embedding (Transformer 방식)으로 $t$를 feature space에 투영
	•	U-Net 내부 residual block에 이 임베딩을 추가해 모든 layer가 $t$에 조건화되도록 함

어려웠던 부분은: embedding이 어떻게 skip connection을 따라 흐르고, 어떤 방식으로 feature map과 작용하는지 직관적으로 떠올리기 어려웠음

4.3 U-Net 구조와 skip connection
	•	U-Net은 pixelCNN++ 기반 구조를 채택
	•	down block은 convolution + pooling → 이해 쉬움
	•	up block은 transpose conv + skip connection
	•	skip connection은 같은 resolution의 down block feature를 concat → semantic 정보를 보존하면서 복원에 사용

내가 겪은 어려움: skip connection이 ε 예측에서 어떤 역할을 하는지 수학적으로 보긴 했지만, 직관적 의미를 파악하는 데 시간이 걸렸음

⸻

🔄 5. 샘플링 알고리즘 분석 (Algorithm 2)

# Sampling pseudocode
x_T ~ N(0, I)
for t = T to 1:
    x_{t-1} = (1/√alpha_t)(x_t - (1 - alpha_t)/√(1 - \bar{alpha}_t) * ε_θ(x_t, t)) + σ_t * z

	•	마치 Langevin dynamics처럼 작동함
	•	큰 특징부터 먼저 복원되고 점차 디테일이 채워짐 (coarse-to-fine)

Figure 6, 10을 통해 시각적으로 확인 가능함

⸻

🧪 6. 실험 요약 (Section 4)
	•	CIFAR10에서 FID 3.17, Inception Score 9.46 (state-of-the-art)
	•	실험적으로도 $\epsilon$-prediction이 $\mu$-prediction보다 품질 우수함을 확인
	•	Reverse process variance를 학습시키는 경우 학습이 불안정해짐

⸻

💡 7. 이 논문이 나에게 준 관점
	•	수식이 단순해 보여도, 각 구성요소는 매우 정교한 목적을 위해 설계됨
	•	“왜 이런 parameterization을 썼을까?“라는 관점으로 수식을 바라보면 이해도가 훨씬 올라감
	•	단순한 재현이 아닌 직관과 해석을 기반으로 연구자적 사고 훈련이 가능함

⸻

📚 참고 문헌
	•	[55] Yang Song & Stefano Ermon. Score-based generative models
	•	[53] Jascha Sohl-Dickstein et al., Deep Unsupervised Learning Using Nonequilibrium Thermodynamics
	•	[61] Pascal Vincent, Score matching and denoising autoencoders
