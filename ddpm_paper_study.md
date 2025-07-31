### 📎 DDPM이 latent variable model인 이유

일반적인 latent variable model은 다음과 같은 구조를 가짐:

\\[
p(x) = \\int p(x|z) p(z) dz
\\]

DDPM도 다음과 같이 표현됨:

\\[
p_\\theta(x_0) = \\int p_\\theta(x_{0:T}) dx_{1:T}
\\]
