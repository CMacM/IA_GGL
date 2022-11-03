import numpy as np
import pyccl as ccl
from importlib import reload
import scipy

# Use Danielle's code to get parameters for Halo model
import params_LSST_DESI as pa
import shared_functions_wlp_wls as ws
from scipy.special import erf

import LSST_redshifts as zed
reload(zed)

cosmo = zed.cosmo
survey_year = zed.survey_year

# We will use a mass definition with Delta = 200 times the matter density
hmd_200m = ccl.halos.MassDef200m()

# The Bhattacharya 2013 concentration-mass relation (used in LSST SRD)
cM = ccl.halos.ConcentrationBhattacharya13(hmd_200m)

# The Tinker 2008 mass function
nM = ccl.halos.MassFuncTinker08(cosmo, mass_def=hmd_200m)

# The Tinker 2010 halo bias
bM = ccl.halos.HaloBiasTinker10(cosmo, mass_def=hmd_200m)

# The NFW profile to characterize the matter density around halos
pM = ccl.halos.profiles.HaloProfileNFW(cM)

#################### Classes for custom HODs for sources and lenses ####################

class LensHOD(ccl.halos.HaloProfileNFW):
    def __init__(self, c_M_relation,
                 lMmin=12.02, lMminp=-1.34,
                 lM0=6.6, lM0p=-1.43,
                 lM1=13.27, lM1p=-0.323):
        self.lMmin=lMmin
        self.lMminp=lMminp
        self.lM0=lM0
        self.lM0p=lM0p
        self.lM1=lM1
        self.lM1p=lM1p
        self.a0 = 1./(1+0.65)
        self.sigmaLogM = 0.4
        self.alpha = 1.
        super(LensHOD, self).__init__(c_M_relation)
        self._fourier = self._fourier_analytic_hod
    
    def _Nc(self, M, a):
        # Number of centrals
        # try with model used in notebook, this could be more up to date
        Mmin = 10.**self._lMmin(a)
        return 0.5 * (1 + erf(np.log(M / Mmin) / self.sigmaLogM))
        
        #return ws.get_Ncen_More(M, 'LSST_DESI')

    def _Ns(self, M, a):
        # Number of satellites
        # try with model used in notebook
        M0 = 10.**self._lM0(a)
        M1 = 10.**self._lM1(a)
        return np.heaviside(M-M0,1) * ((M - M0) / M1)**self.alpha

        #return ws.get_Ncen_More(M, 'LSST_DESI')

    def _lMmin(self, a):
        return self.lMmin + self.lMminp * (a - self.a0)

    def _lM0(self, a):
        return self.lM0 + self.lM0p * (a - self.a0)

    def _lM1(self, a):
        return self.lM1 + self.lM1p * (a - self.a0)

    def _fourier_analytic_hod(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)
        
        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        
        # NFW profile
        uk = self._fourier_analytic(cosmo, k_use, M_use, a, mass_def) / M_use[:, None]

        prof = Nc[:, None] * (1 + Ns[:, None] * uk)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_variance(self, cosmo, k, M, a, mass_def):
        # Fourier-space variance of the HOD profile
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        
        # NFW profile
        uk = self._fourier_analytic(cosmo, k_use, M_use, a, mass_def) / M_use[:, None]

        prof = Ns[:, None] * uk
        prof = Nc[:, None] * (2 * prof + prof**2)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
    
# create custom halo profile class for sources       
class SourceHOD(ccl.halos.HaloProfileNFW):
    def __init__(self, c_M_relation,
                 lMmin=12.02, lMminp=-1.34,
                 lM0=6.6, lM0p=-1.43,
                 lM1=13.27, lM1p=-0.323):
        self.lMmin=lMmin
        self.lMminp=lMminp
        self.lM0=lM0
        self.lM0p=lM0p
        self.lM1=lM1
        self.lM1p=lM1p
        self.a0 = 1./(1+0.65)
        self.sigmaLogM = 0.4
        self.alpha = 1.
        super(SourceHOD, self).__init__(c_M_relation)
        self._fourier = self._fourier_analytic_hod
    
    def _Nc(self, M, a):
        # Number of centrals
        tot_nsrc= get_vol_dens(fsky=pa.fsky, N=pa.N_shapes)
        Mstarlow = get_Mstarlow(ngal=tot_nsrc)

        return ws.get_Ncen_Zu(Mh=M, Mstar=Mstarlow, survey='LSST_DESI')

    def _Ns(self, M, a):
        # Number of satellites
        tot_nsrc= get_vol_dens(fsky=pa.fsky, N=pa.N_shapes)
        Mstarlow = get_Mstarlow(ngal=tot_nsrc)

        return ws.get_Nsat_Zu(M_h=M, Mstar=Mstarlow, case='tot', survey='LSST_DESI')

    def _lMmin(self, a):
        return self.lMmin + self.lMminp * (a - self.a0)

    def _lM0(self, a):
        return self.lM0 + self.lM0p * (a - self.a0)

    def _lM1(self, a):
        return self.lM1 + self.lM1p * (a - self.a0)

    def _fourier_analytic_hod(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)
        
        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        
        # NFW profile
        uk = self._fourier_analytic(cosmo, k_use, M_use, a, mass_def) / M_use[:, None]

        prof = Nc[:, None] * (1 + Ns[:, None] * uk)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_variance(self, cosmo, k, M, a, mass_def):
        # Fourier-space variance of the HOD profile
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        
        # NFW profile
        uk = self._fourier_analytic(cosmo, k_use, M_use, a, mass_def) / M_use[:, None]

        prof = Ns[:, None] * uk
        prof = Nc[:, None] * (2 * prof + prof**2)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

# Halo Occupation Distribution model    
Lpg = LensHOD(cM)
Spg = SourceHOD(cM)

# set custom class to calculate non-trivial 2pt cumulant
class Profile2ptHOD(ccl.halos.Profile2pt):
    def fourier_2pt(self, prof, cosmo, k, M, a,
                          prof2=None, mass_def=None):
        return prof._fourier_variance(cosmo, k, M ,a, mass_def)

# 2pt cumulant of HOD
HOD2pt = Profile2ptHOD()

# Halo model calculator for computing mass integrals
hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd_200m)

# calculates galaxy-matter power spectrum
def get_1D_power(corr, k_arr, plot_fig='n', save_fig='n'):
    '''Calculate power spectra as function of k'''
    
    # ccl by default computes one and two halo terms
    
    # Galaxy-Matter
    if corr == 'gm': 
        pk = ccl.halos.halomod_power_spectrum(cosmo, hmc, k_arr, 1.,
                                             Lpg, prof2=pM, normprof1=True, normprof2=True)
    # Galaxy-Galaxy
    elif corr == 'gg':
        # 2-point cumulant controls the 1 halo term
        class Profile2ptHOD(ccl.halos.Profile2pt):
            def fourier_2pt(self, prof, cosmo, k, M, a,
                          prof2, mass_def=None):
                return prof._fourier_variance(cosmo, k, M ,a, mass_def)
        HOD2pt = Profile2ptHOD()

        pk = ccl.halos.halomod_power_spectrum(cosmo, hmc, k_arr, 1.,
                                             Lpg, prof_2pt=HOD2pt, prof2=Spg, 
                                              normprof1=True, normprof2=True)
    # Matter-Matter
    elif corr == 'mm':
        pk = ccl.halos.halomod_power_spectrum(cosmo, hmc, k_arr, 1.,
                                         pM, normprof1=True)

    return pk


def get_Pk2D(corr, k_arr, a_arr):
    '''Calculate 2D power spectrum from custome HOD and NFW profile'''
    
    # Galaxy-Matter
    if corr == 'gm':
        pk2D = ccl.halos.halomod_Pk2D(cosmo, hmc, Lpg, prof2=pM, 
                                    normprof1=True, normprof2=True, 
                                    lk_arr=np.log(k_arr), a_arr=a_arr)
    # Galaxy-Galaxy
    elif corr == 'gg':    
        pk2D = ccl.halos.halomod_Pk2D(cosmo, hmc, Lpg, prof_2pt=HOD2pt, prof2=Spg, 
                                      normprof1=True, normprof2=True, 
                                      lk_arr=np.log(k_arr), a_arr=a_arr)
    #Matter-Matter
    elif corr == 'mm':
        pk_2D = ccl.halos.halomod_Pk2D(cosmo, hmc, pM,
                                normprof1=True,
                                lk_arr=np.log(k_arr), a_arr=a_arr)
    
    return pk2D

def get_vol_dens(fsky, N, year=survey_year):
    
    z_s, dndz_s, zseff = zed.get_dndz_spec('source', year)
	
    # Get dNdz, normalized to the number of source galaxies N
    norm = scipy.integrate.simps(dndz_s, z_s)
    dNdz_num = N * dndz_s / norm 
	
    # Get factors needed to change to n(z)
    # Use the source HOD cosmological parameters here to be consistent
    OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
    H_over_c = pa.HH0 * ( (pa.OmC+pa.OmB)*(1.+z_s)**3 + OmL + (pa.OmR+pa.OmN) * (1.+z_s)**4 )**(0.5)
	
    # volume density as a function of z
    # UPDATE CCL - use CCL comoving distance?
    chi = ccl.comoving_radial_distance(zed.cosmo, 1./(1.+z_s)) * (pa.HH0 / 100.) # CCL returns in Mpc but we want Mpc/h
    ndens_ofz = dNdz_num * H_over_c / ( 4. * np.pi * fsky * chi**2 )
	
    # We want to integrate this over the window function of lenses x sources, because that's the redshift range on which we care about the number density:
    z_win, win = zed.window(year)
    interp_ndens = scipy.interpolate.interp1d(z_s, ndens_ofz)
    ndens_forwin = interp_ndens(z_win)
	
    ndens_avg = scipy.integrate.simps(ndens_forwin * win, z_win)
	
    return ndens_avg

def get_Mstarlow(ngal, year=survey_year):
    
    # Get the window function for the lenses x sources
    z_l, win = zed.window(year)
	
    # Use the HOD model from Zu & Mandelbaum 2015
		
    # Define a vector of Mstar_low value to try
    Ms_low_vec = np.logspace(7., 10.,200)
    # Define a vector of Mh values to integrate over
    Mh_vec = np.logspace(9., 16., 100)
	
    # Get Nsat and Ncen as a function of the values of the two above arrays
    Nsat = ws.get_Nsat_Zu(Mh_vec, Ms_low_vec, 'tot', 'LSST_DESI') # Get the occupation number counting all sats in the sample.
    Ncen = ws.get_Ncen_Zu(Mh_vec, Ms_low_vec, 'LSST_DESI')
    
    # Get the halo mass function (from CCL) to integrate over (dn / dlog10M, Tinker 2010 )
    HMF = np.zeros((len(Mh_vec), len(z_l)))
    #print("In Mstar low - check units of HMF!")
    for zi in range(0,len(z_l)):
        #print "zi=", zi
        HMF_class = ccl.halos.MassFuncTinker10(cosmo)
        HMF[:,zi] = HMF_class.get_mass_function(cosmo, Mh_vec / (pa.HH0/100.), 1./ (1. + z_l[zi]))
	
    # Now get what nsrc should be for each Mstar_low cut 
    nsrc_of_Mstar_z = np.zeros((len(Ms_low_vec), len(z_l)))
    for msi in range(0,len(Ms_low_vec)):
        #print "Msi=", i
        for zi in range(0,len(z_l)):
            nsrc_of_Mstar_z[msi, zi] = scipy.integrate.simps(HMF[:, zi] * ( Nsat[msi, :] + Ncen[msi, :]), np.log10(Mh_vec / (pa.HH0 / 100.))) / (pa.HH0 / 100.)**3
	
    # Integrate this over the z window function
    nsrc_of_Mstar = np.zeros(len(Ms_low_vec))
    for i in range(0,len(Ms_low_vec)):
        nsrc_of_Mstar[i] = scipy.integrate.simps(nsrc_of_Mstar_z[i, :] * win, z_l)
	
    # Get the correct Mstar cut
    try:
        ind = next(j[0] for j in enumerate(nsrc_of_Mstar) if j[1]<=ngal)
    except StopIteration:
        pass
    
    Mstarlow = Ms_low_vec[ind]
	
    return Mstarlow
