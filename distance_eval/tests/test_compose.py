"""Sanity: compose_l2 matches brute-force on tiny PQ reconstruction."""

from __future__ import annotations

import numpy as np

from distance_eval.base import compose_l2


def test_compose_l2_matches_bruteforce_pq_style():
    np.random.seed(0)
    nq, nb, M, dsub = 4, 5, 3, 2
    dim = M * dsub
    ksub = 4
    q = np.random.randn(nq, dim).astype(np.float32)
    centroids = np.random.randn(M, ksub, dsub).astype(np.float32)
    codes = np.random.randint(0, ksub, size=(nb, M), dtype=np.int32)

    recon = np.zeros((nb, dim), dtype=np.float32)
    for j in range(M):
        recon[:, j * dsub : (j + 1) * dsub] = centroids[j][codes[:, j]]

    # sqeuclidean cdist: (nq, nb)
    diff = q[:, None, :] - recon[None, :, :]
    exact = np.sum(diff * diff, axis=2).astype(np.float32)
    ip = (q @ recon.T).astype(np.float32)
    qn = np.sum(q * q, axis=1).astype(np.float32)
    dn = np.sum(recon * recon, axis=1).astype(np.float32)
    adc = compose_l2(qn, dn, ip)
    np.testing.assert_allclose(adc, exact, rtol=1e-4, atol=1e-3)
