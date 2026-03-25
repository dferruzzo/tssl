import matplotlib.pyplot as plt
import numpy as np

from tssl import fourier_series

# x(t): sinal quadrado de entrada
A = 1
TH = 2
TL = 2
T0 = TH + TL
w0 = 2 * np.pi / T0
N = 5000

X0 = A * TH / T0
Xk = lambda k: (1j * A / (2 * k * np.pi)) * (np.exp(-1j * k * w0 * TH) - 1)
sol_x = fourier_series(X0, Xk, w0, N)

# Resposta em frequência (RC)
R = 1000
C = 0.5 * 1e-3
Hv = lambda k: 1 / (1 + 1j * k * w0 * R * C)

X0v = X0 * Hv(0)
Xkv = lambda k: Xk(k) * Hv(k)
sol_vc = fourier_series(X0v, Xkv, w0, N)

plt.figure()
plt.plot(sol_x["t"], sol_x["x"].real, linewidth=3, label="x(t)")
plt.plot(sol_vc["t"], sol_vc["x"].real, linewidth=3, label="v_c(t)")
plt.grid(True)
plt.xlabel("Tempo")
plt.legend()
plt.title(f"N = {N}")
plt.show()
