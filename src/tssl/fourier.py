from __future__ import annotations

from typing import Callable

import numpy as np


def fourier_series(
    X0: complex,
    Xk: Callable[[int], complex],
    w0: float,
    N: int,
):
    """Calcula a Série de Fourier (forma exponencial) e retorna amostras.

    Parâmetros
    - X0: termo constante da série
    - Xk: função que retorna o k-ésimo coeficiente complexo Xk(k)
    - w0: frequência angular fundamental
    - N: número de harmônicos (k = 0..N-1)

    Retorno
    - dict com vetores/escalares úteis (t, x, w0, T0, dt, etc.)
    """

    a0 = X0
    ak = lambda k: 2 * np.real(Xk(k))
    bk = lambda k: -2 * np.imag(Xk(k))

    c0 = a0
    ck = lambda k: 2 * np.abs(Xk(k))
    thetak = lambda k: -np.arctan2(bk(k), ak(k))

    T0 = 2 * np.pi / w0
    t0 = 0
    tf = 2 * T0
    amostras = 100
    dt = T0 / amostras
    t = np.linspace(t0, tf, 2 * amostras)
    x = np.ones(len(t)) * c0

    cks: list[complex] = [c0]
    thetaks: list[float] = [float(np.angle(c0))]

    for k in range(1, N):
        cks.append(ck(k))
        thetaks.append(float(thetak(k)))
        x += ck(k) * np.cos(k * w0 * t + thetak(k))

    return {
        "t": t,
        "x": x,
        "w0": w0,
        "T0": T0,
        "dt": dt,
        "c0'": c0,
        "ck'": np.array(cks),
        "thetak'": np.array(thetaks),
        "N": N,
        "k": np.arange(0, N, 1),
    }
