import numpy as np

from ion_phys.rate_equations import Rates


class OBEMatrices:
    """Matrices for optical bloch equation calculations

    Current implementation uses dense matrices. The fill ratio is ~25%,
    hence sparse matrices may provide some benefit.

    Currently only first order laser effects are included

    All time evolution is calculated in the rotating wave approximation.

    Time evolution can may be calculated in two different pictures:
        cmat: density matrix with free ion phase accounted for. This is
            generally valid.
        smat: density matrix with free ion and laser detuning phase evolution
            accounted for. This is only valid if there are no closed coherently
            driven transition cycles with a net detuning.
            Unlike cdot, sdot has time translation symmetry.
            This simplifies numerical integration.

    definitions follow Hugo thesis:
    https://www2.physics.ox.ac.uk/sites/default/files/2011-08-15/
    hajanacekthesis25aprilone_pdf_17569.pdf
    """
    def __init__(self, ion):
        self.rates = Rates(ion)
        # spont_mat.T contains A_ij (and -GammaJ on diagonal)
        self.spont_mat = self.rates.get_spont()
        # delta in units: rad s^-1 (delta_mat[upper, lower]>0.)
        self.delta_mat = self.rates.ion.delta(
            np.arange(self.rates.ion.num_states)[np.newaxis, :],
            np.arange(self.rates.ion.num_states)[:, np.newaxis])

    def get_cdot(self, cmat, t, lasers):
        """get time evolution of the c-matrix

        cmat: ion density-matrix with eigenstate precession subtracted out
        t: time (evolves phases from laser detuning; @t=0 all phases are 0)
        lasers: iterable of lasers interacting with ion.

        return: cdot (time deriviative of c-matrix)
        """
        cdot = np.zeros(self.spont_mat.shape, dtype=np.complex128)
        for l in lasers:
            cdot += self._get_laser_cdot(cmat, t, l)
        cdot += self._get_spon_cdot(cmat)
        return cdot

    def _get_laser_cdot(self, cmat, t, laser):
        """cdot terms from single laser"""
        rmat = self._get_laser_rmat(laser)
        phase_mat = np.exp(1j * t * self._get_detuning_mat(laser))
        cdot = 1j * np.einsum('lk, jl, jl -> jk', cmat, rmat, phase_mat)
        cdot += -1j * np.einsum('jl, lk, lk -> jk', cmat, rmat, phase_mat)
        return cdot

    def _get_spon_cdot(self, cmat):
        """cdot terms from spontaneous emission"""
        ion = self.rates.ion
        spon = ion.GammaJ[:, np.newaxis] + ion.GammaJ[np.newaxis, :]

        cdot = -0.5 * spon * cmat  # off diagonal
        # diagonal - note spont_mat.T contains A_ij (and -GammaJ on diagonal)
        diag = np.einsum('lj, ll -> j', self.spont_mat.T, cmat)
        np.fill_diagonal(cdot, diag)  # in place!
        return cdot

    def _get_laser_rmat(self, laser):
        """Definition of Rabi frequency from #10, equation 29"""
        ion = self.rates.ion
        trans = ion.transitions[laser.transition]
        gamma_tot = ion.GammaJ[ion.levels[trans.upper]._start_ind]
        detuning_mat = self._get_detuning_mat(laser)
        stim_rates = self.rates.get_stim([laser])
        np.fill_diagonal(stim_rates, 0.)  # in place!

        rmat = np.sqrt(stim_rates / gamma_tot
                       * (detuning_mat**2 + 0.25 * gamma_tot**2))
        return rmat  # = omega/2

    def _get_detuning_mat(self, laser):
        """Matrix encoding laser detuning from hyperfine transitions

        This matrix is anti-symmetric.
        defined such that for a blue detuned laser:
            detuning_mat[upper, lower] > 0.0

        returns: detuning_mat"""
        ion = self.rates.ion
        trans = ion.transitions[laser.transition]
        lower, upper = ion.slice(trans.lower), ion.slice(trans.upper)

        detuning_mat = np.zeros(self.spont_mat.shape, dtype=np.float64)
        detuning_mat[upper, lower] = laser.delta - self.delta_mat[upper, lower]
        detuning_mat -= detuning_mat.T
        return detuning_mat

    def cmat_to_density_mat(self, cmat, t=0):
        return cmat * np.exp(-1j * t * self.delta_mat)

    def density_mat_to_cmat(self, density_mat, t=0):
        return density_mat * np.exp(1j * t * self.delta_mat)
