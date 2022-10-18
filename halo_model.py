import numpy as np
import pyccl as ccl

# Use Danielle's code to get parameters for Halo model
import params_LSST_DESI as pa
import shared_functions_wlp_wls as ws

# Start by initialsising cosmology
while 1==1:
    answer = input('halo_model: Initialised with Omega_c=%g, Omega_b=%g, h=%g, sigma8=%g, n_s=%g. Modify cosmology? Input y or n.'%(pa.OmC, pa.OmB, pa.HH0/100., pa.sigma8, pa.n_s_cosmo))
    if str(answer) == 'y':
        Omega_c = float(input('Input Omega_c'))
        Omega_b = float(input('Input Omega_b'))
        h = float(input('Input h'))
        sigma8 = float(input('Input sigma8'))
        n_s = float(input('Input n_s'))
        cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s)
        print('Values set to: Omega_c=%g, Omega_b=%g, h=%g, sigma8=%g, n_s=%g'%(Omega_c, Omega_b, h, sigma8, n_s))
        break
    elif str(answer) == 'n':
        cosmo = ccl.Cosmology(Omega_c=pa.OmC, Omega_b=pa.OmB, h=(pa.HH0/100.), sigma8=pa.sigma8, n_s=pa.n_s_cosmo)
        break
    else:
        continue

# We will use a mass definition with Delta = 200 times the matter density
hmd_200m = ccl.halos.MassDef200m()

# The Duffy 2008 concentration-mass relation
cM = ccl.halos.ConcentrationDuffy08(hmd_200m)

# The Tinker 2008 mass function
nM = ccl.halos.MassFuncTinker08(cosmo, mass_def=hmd_200m)

# The Tinker 2010 halo bias
bM = ccl.halos.HaloBiasTinker10(cosmo, mass_def=hmd_200m)

# The NFW profile to characterize the matter density around halos
pM = ccl.halos.profiles.HaloProfileNFW(cM)
        
# create custom halo profile class for use later        
class HaloProfileHOD(ccl.halos.HaloProfileNFW):
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
        super(HaloProfileHOD, self).__init__(c_M_relation)
        self._fourier = self._fourier_analytic_hod
    
    def _Nc(self, M, a):
        # Number of centrals

        return ws.get_Ncen_More(M, 'LSST_DESI')
    
    def _Ns(self, M, a):
        # Number of satellites

        return ws.get_Nsat_More(M, 'LSST_DESI')

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
pg = HaloProfileHOD(cM)

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
                                             pg, prof2=pM, normprof1=True, normprof2=True)
    # Galaxy-Galaxy
    elif corr == 'gg':
        # 2-point cumulant controls the 1 halo term
        class Profile2ptHOD(ccl.halos.Profile2pt):
            def fourier_2pt(self, prof, cosmo, k, M, a,
                          prof2=None, mass_def=None):
                return prof._fourier_variance(cosmo, k, M ,a, mass_def)
        HOD2pt = Profile2ptHOD()

        pk = ccl.halos.halomod_power_spectrum(cosmo, hmc, k_arr, 1.,
                                             pg, prof_2pt=HOD2pt,
                                             normprof1=True)
    # Matter-Matter
    elif corr == 'mm':
        pk = ccl.halos.halomod_power_spectrum(cosmo, hmc, k_arr, 1.,
                                         pM, normprof1=True)

    return pk


def get_Pk2D(corr, k_arr, a_arr):
    '''Calculate 2D power spectrum from custome HOD and NFW profile'''
    
    # Galaxy-Matter
    if corr == 'gm':
        pk2D = ccl.halos.halomod_Pk2D(cosmo, hmc, pg, prof2=pM, 
                                    normprof1=True, normprof2=True, 
                                    lk_arr=np.log(k_arr), a_arr=a_arr)
    # Galaxy-Galaxy
    elif corr == 'gg':    
        pk2D = ccl.halos.halomod_Pk2D(cosmo, hmc, pg, prof_2pt=HOD2pt,
                                normprof1=True,
                                lk_arr=np.log(k_arr), a_arr=a_arr)
    #Matter-Matter
    elif corr == 'mm':
        pk_2D = ccl.halos.halomod_Pk2D(cosmo, hmc, pM,
                                normprof1=True,
                                lk_arr=np.log(k_arr), a_arr=a_arr)
    
    return pk2D
