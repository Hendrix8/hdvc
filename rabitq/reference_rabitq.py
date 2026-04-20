# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Vendored from lib/faiss/tests/test_rabitq.py (reference implementation).

import numpy as np


def random_rotation(d, seed=123):
    rs = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rs.randn(d, d))
    return Q


class ReferenceRabitQ:
    """Exact translation of the RaBitQ paper (quantizer + stored codes)."""

    def __init__(self, d, Bq=4):
        self.d = d
        self.Bq = Bq

    def train(self, xtrain, P):
        self.centroid = xtrain.mean(0)
        self.P = P

    def rotation(self, x):
        return x @ self.P

    def inv_rotation(self, x):
        return x @ self.P.T

    def add(self, Or):
        Orc = Or - self.centroid
        self.O_norms = np.sqrt((Orc**2).sum(1))
        O = Orc / self.O_norms[:, None]

        self.Xbarb = (self.inv_rotation(Orc) > 0).astype("int8")
        Obar = self.rotation((2 * self.Xbarb - 1) / np.sqrt(self.d))
        self.o_Obar = (O * Obar).sum(1)

    def distances(self, Qr):
        d = self.d
        Bq = self.Bq

        Qrc = Qr - self.centroid
        Qrc_norms = np.sqrt((Qrc**2).sum(1))[:, None]
        Qprime = self.inv_rotation(Qrc)

        mins, maxes = Qprime.min(axis=1)[:, None], Qprime.max(axis=1)[:, None]
        Delta = (maxes - mins) / (2**Bq - 1)

        qbar = np.round((Qprime - mins) / Delta)
        dp = (qbar[:, None, :] * self.Xbarb[None, :, :]).sum(2)

        sum_X = self.Xbarb.sum(1)
        sum_Q = qbar.sum(1)[:, None]
        sD = np.sqrt(d)
        xbar_qbar = 2 * Delta / sD * dp
        xbar_qbar += 2 * mins / sD * sum_X
        xbar_qbar -= Delta / sD * sum_Q
        xbar_qbar -= sD * mins

        q_o = xbar_qbar / self.o_Obar
        dis2_q_o = self.O_norms**2 + Qrc_norms**2 - 2 * self.O_norms * q_o

        return dis2_q_o
