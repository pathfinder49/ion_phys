from ion_phys.obe import OBEMatrices
from ion_phys.ions.ca43 import Ca43, ground_level, P12, D32, P32, D52
from ion_phys import Laser
from ion_phys.rate_equations import Rates

import unittest
import numpy as np


class TestOBE(unittest.TestCase):
    def setUp(self):
        ion = Ca43(B=146e-4, level_filter=[ground_level, P12, D32, P32, D52])
        delta0 = ion.delta(ion.index(ground_level, 4), ion.index(P12, +4))
        delta1 = ion.delta(ion.index(ground_level, 3, F=3), ion.index(P12, +4))
        delta2 = ion.delta(ion.index(D32, +5), ion.index(P12, +4, F=4))
        self.lasers = [Laser("397", q=0, I=50, delta=delta0 - 1e7),
                       Laser("397", q=+1, I=50, delta=delta0 - 1e7),
                       Laser("397", q=0, I=50, delta=delta1 - 1e7),
                       Laser("397", q=+1, I=50, delta=delta1 - 1e7),
                       Laser("866", q=-1, I=1e3, delta=delta2 - 1e7),
                       Laser("866", q=0, I=1e3, delta=delta2 - 1e7),
                       Laser("854", q=-1, I=1e2, delta=0),
                       Laser("854", q=0, I=1e2, delta=0),
                       ]
        self.ion = ion
        self.delta0 = delta0
        self.rates = Rates(ion)
        self.obe = OBEMatrices(ion)

    def test_detuning_mat(self):
        "Detuning matrix as defined in Hugo 2.12"
        laser = Laser("397", q=0, I=100, delta=self.delta0)
        det_mat = self.obe._get_detuning_mat(laser)
        # detuning matrix is anti-symmetric
        self.assertTrue(np.all(det_mat == -det_mat.T))
        # applied laser is resonant
        self.assertEqual(det_mat[self.ion.index(P12, +4),
                                 self.ion.index(ground_level, 4)],
                         0.0)
        # blue detuned laser should have +ve detuning for [upper, lower]
        self.assertGreater(det_mat[self.ion.index(P12, +4),
                                   self.ion.index(ground_level, 3, F=3)],
                           0.0)
        # red detuned laser should have -ve detuning for [upper, lower]
        self.assertLess(det_mat[self.ion.index(P12, +4),
                                self.ion.index(ground_level, 3, F=4)],
                        0.0)

    def test_laser_rmat(self):
        laser = Laser("397", q=0, I=1, delta=self.delta0)
        rmat = self.obe._get_laser_rmat(laser)
        # symmetric
        self.assertTrue(np.all(rmat == rmat.T))
        # zero on diagonal
        self.assertTrue(np.all(np.diag(rmat) == 0))
        # zero for incorrect polerisation
        self.assertEqual(
            rmat[self.ion.index(P12, +4),
                 self.ion.index(ground_level, 3, F=4)],
            0.0)
        # equal half rabi frequency on selected transition
        self.assertEqual(
            rmat[self.ion.index(P12, +4),
                 self.ion.index(ground_level, 4, F=4)],
            0.5 * laser.I * np.sqrt(
                self.obe.spont_mat.T[self.ion.index(P12, +4),
                                     self.ion.index(ground_level, 4, F=4)]
                * self.ion.GammaJ[self.ion.index(P12, +4)]))
        # zero for I=0
        laser = Laser("397", q=0, I=0.0, delta=self.delta0)
        rmat = self.obe._get_laser_rmat(laser)
        self.assertEqual(rmat[self.ion.index(P12, +4),
                              self.ion.index(ground_level, 4, F=4)],
                         0.0)

    def test_get_spon_cdot(self):
        cmat = np.zeros(self.obe.spont_mat.shape)
        idx = self.ion.index(P12, 4, F=4)
        cmat = np.zeros(self.obe.spont_mat.shape)
        cmat[idx, idx] = 1.0
        cdot = self.obe._get_spon_cdot(cmat)
        # P state decay rate
        self.assertEqual(cdot[idx, idx], -self.ion.GammaJ[idx])

    def test_get_cdot(self):
        laser0 = Laser("397", q=0, I=0, delta=self.delta0)
        laser1 = Laser("397", q=0, I=1, delta=self.delta0)
        cmat = np.zeros(self.obe.spont_mat.shape)
        idx = self.ion.index(ground_level, 4, F=4)
        cmat[idx, idx] = 1.0
        cdot0 = self.obe.get_cdot(cmat, t=0.1, lasers=[laser0])
        cdot1 = self.obe.get_cdot(cmat, t=0.1, lasers=[laser1])
        # ground state does not decay
        self.assertTrue(np.all(np.zeros(cdot0.shape) == cdot0))
        # cdot is hermitian
        self.assertTrue(np.all(cdot1.conj().T == cdot1))
        # driven transition
        self.assertNotEqual(
            np.abs(cdot1[self.ion.index(P12, +4),
                         self.ion.index(ground_level, 4, F=4)]),
            0.0)
        cdot1[self.ion.index(P12, +4),
              self.ion.index(ground_level, 4, F=4)] = 0.
        cdot1[self.ion.index(ground_level, 4, F=4),
              self.ion.index(P12, +4)] = 0.
        # no other dynamics!
        self.assertTrue(np.all(np.zeros(cdot1.shape) == cdot1))


if __name__ == '__main__':
    unittest.main()
