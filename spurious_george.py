import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import shared_functions_setup as setup
import shared_functions_wlp_wls as ws
import os.path
import pyccl as ccl
import params_LSST_DESI as pa
#import main.py as main

'''Objective is to use script as a basis for computing the effects of spurious signals'''

# Start by initialsising cosmology
while 1==1:
    answer = input('Initialised with Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96. Modify cosmology? Input y or n.')
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
        cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96)
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
        
pg = HaloProfileHOD(cM)

# Halo model calculator for computing mass integrals
hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd_200m)

# calculates galaxy-matter power spectrum
def get_gM_power(k_arr, save_fig='n'):
    '''Calculate Galaxy-Galaxy power spectrum using custom halo profile and display result'''
    
    # ccl by default computes one and two halo terms 
    pk_gM = ccl.halos.halomod_power_spectrum(cosmo, hmc, k_arr, 1.,
                                         pg, prof2=pM,
                                         normprof1=True, normprof2=True)
    
    plt.figure(figsize=([8,5]))
    plt.plot(k_arr, pk_gM, label='$P_{gM}(k)$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=16)
    plt.ylabel(r'$P(k)\,\,[{\rm Mpc}^3]$', fontsize=15)
    plt.xlabel(r'$k\,\,[{\rm Mpc}^{-1}]$', fontsize=15);
    
    if save_fig == 'y':
        plt.savefig('gM_power_spec.png')
    elif save_fig == 'n':
        pass
    
    return pk_gM

# simply pulls spec-z distributions and z values from LSST forecast data release
# distributions should already be normalised
def get_dNdz_spec(gtype, year):
    '''Get redshift data from LSST forecast data files'''
    
    if year == 1 and gtype == 'source':
        zdata = np.loadtxt('/home/b7009348/WGL_project/LSST-forecast-data/LSST_DESC_SRD_v1_release/forecasting/WL-LSS-CL/zdistris/zdistri_model_z0=1.300000e-01_beta=7.800000e-01_Y1_source', usecols=[1,3])
        z_arr = zdata[:,0]
        dNdz = zdata[:,1]
    elif year == 10 and gtype == 'source':
        zdata = np.loadtxt('/home/b7009348/WGL_project/LSST-forecast-data/LSST_DESC_SRD_v1_release/forecasting/WL-LSS-CL/zdistris/zdistri_model_z0=1.100000e-01_beta=6.800000e-01_Y10_source', usecols=[1,3])
        z_arr = zdata[:,0]
        dNdz = zdata[:,1]
    elif year == 1 and gtype == 'lens':
        zdata = np.loadtxt('/home/b7009348/WGL_project/LSST-forecast-data/LSST_DESC_SRD_v1_release/forecasting/WL-LSS-CL/zdistris/zdistri_model_z0=2.600000e-01_beta=9.400000e-01_Y1_lens', usecols=[1,3])
        z_arr = zdata[:,0]
        dNdz = zdata[:,1]
    elif year == 10 and gtype == 'lens':
        zdata = np.loadtxt('/home/b7009348/WGL_project/LSST-forecast-data/LSST_DESC_SRD_v1_release/forecasting/WL-LSS-CL/zdistris/zdistri_model_z0=2.800000e-01_beta=9.000000e-01_Y10_lens', usecols=[1,3])
        z_arr = zdata[:,0]
        dNdz = zdata[:,1]
    else:
        print('Not an LSST release year')
        z_arr, dNdz = 0.
        
    return z_arr, dNdz


def get_dNdz_phot(gtype, year, plot_fig='n', save_fig='n'):
    
    # get spectroscopic data and set arbitrary photo-z points in extended
    # redshift range
    z_s, dNdz_s = get_dNdz_spec(gtype, year)
    z_ph = np.linspace(np.min(z_s), np.max(z_s)+0.5, 300)

    if gtype == 'source':
        sig_z = 0.05*(1. + z_s)
    elif gtype == 'lens':
        sig_z = 0.03*(1. + z_s)
    
    # find probability of source with z_ph having spectroscopic redshift z_s
    p_zs_zph = np.zeros([len(z_s),len(z_ph)])
    for i in range(len(z_ph)):
        p_zs_zph[i,:] =  1. / (np.sqrt(2. * np.pi) * sig_z) * np.exp(-((z_ph - z_s[i])**2) / (2. * sig_z**2))
    
    # integrate over z_s to get dN
    integrand1 = p_zs_zph * dNdz_s
    integral1 = scipy.integrate.simps(integrand1, z_s)
    dN = integral1

    integrand2 = dN
    integral2 = scipy.integrate.simps(integrand2, z_ph)
    dz_ph = integral2

    dNdz_ph = dN / dz_ph
    
    if plot_fig == 'y':
        plt.figure(figsize=[6,5])
        plt.title(gtype, fontsize=15)
        plt.plot(z_ph, dNdz_ph)
        plt.plot(z_s, dNdz_s)
        plt.xlabel('z', fontsize=15)
        plt.ylabel(r'$\frac{dN}{dz}$', fontsize=15)
        plt.legend([r'$z_{ph}$',r'$z_{s}$'], fontsize=16);
    
    if save_fig == 'y':
        plt.savefig('zdist_comparison_post_convolution.png', dpi=300)
        
    return z_ph, dNdz_ph, p_zs_zph
    

def get_LSST_aps(year, ell, k_arr, a_arr, b=1.0, save_fig='n'):
    
    if year == 1:
        lens_z, lens_dNdz = get_dNdz_spec(gtype='lens', year=1)
        source_z, source_dNdz = get_dNdz_spec(gtype='source', year=1)
    elif year == 10:
        lens_z, lens_dNdz = get_dNdz_spec(gtype='lens', year=10)
        source_z, source_dNdz = get_dNdz_spec(gtype='source', year=10)
      
    # init tracers for lenses and sources
    lensTracer = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(lens_z, lens_dNdz), 
                                    bias=(lens_z, b*np.ones(len(lens_z))))
    sourceTracer = ccl.WeakLensingTracer(cosmo, (source_z, source_dNdz))
    
    # create 2D power spectrum objects for galaxy-shear correlation
    pk_gMf = ccl.halos.halomod_Pk2D(cosmo, hmc, pg, prof2=pM, 
                                normprof1=True, normprof2=True, 
                                lk_arr=np.log(k_arr), a_arr=a_arr)
    
    # need to make sure pk_gMf is passed in!
    aps = ccl.angular_cl(cosmo, lensTracer, sourceTracer, ell, p_of_k_a=pk_gMf)
    
    plt.figure(figsize=[6,5])
    plt.plot(ell, aps)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$C_{l}$', fontsize=14)
    plt.xlabel(r'$\ell$', fontsize=14);
    
    if save_fig == 'y':
        plt.savefig('aps_gM_plot.png')
    elif save_fig == 'n':
        pass
    
    return aps
    
def get_gammat(aps, theta_arr, ell, save_fig='n'):
    
    gammat = ccl.correlation(cosmo, ell, aps, theta_arr, type='NG')
    
    plt.figure(figsize=[6,5])
    plt.plot(theta_arr, gammat, linewidth=0, marker='.')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-6, 1e-2])
    plt.ylabel(r'$\gamma_{t}$', fontsize=14)
    plt.xlabel(r'$\theta$ (arcmin)', fontsize=14);
    
    if save_fig == 'y':
        plt.savefig('gammat_plot.png')
    elif save_fig == 'n':
        pass
    
    return gammat

def compute_weights():
    
    weights = 1. / (pa.e_rms_mean**2 + pa.sig_e**2)
    
    return weights

def get_F():
    '''Calculated F from LSST forecast spectroscopic distributions and estimated
    phot-z distribtuion'''
    
    weights = spurious_george.compute_weights()
    year = 1

    z_s, dNdz_s = spurious_george.get_dNdz_spec(gtype='source', year=year)
    z_l, dNdz_l = spurious_george.get_dNdz_spec(gtype='lens', year=year)

    # convolve distributions to estimate dNdz_ph, z_ph is
    # zs_min <= z_s <= zs_max + 0.5
    z_ph, dNdz_ph, p_zs_zph = spurious_george.get_dNdz_phot(gtype='source', year=year)

    #convert source and lens spec-z to com-ra-dist
    chi_lens = ccl.comoving_radial_distance(cosmo, 1./(1. + z_l))
    chi_source = ccl.comoving_radial_distance(cosmo, 1./(1. + z_s))

    #find upper and lower IA lims in com-ra-dist
    chi_up = chi_lens + 100.
    chi_low = chi_lens - 100.

    # remove negative distances from result lower lims
    chi_low = [0. if x < 0. else x for x in chi_low]

    #get scale factors for com-ra-dist lims
    a_up = ccl.scale_factor_of_chi(cosmo, chi_up)
    a_low = ccl.scale_factor_of_chi(cosmo, chi_low)

    #convert scale factors to find lims in redshift space
    z_up = (1./a_up) - 1.
    z_low = (1./a_low) - 1.

    # this gets a little confusing... 

    # calculate integrand for rightmost numerator integral
    integrand1 = dNdz_s * p_zs_zph

    # loop over z_+ and z_- for different z_l and integrate between them
    num_integral1 = np.zeros(np.shape(p_zs_zph))
    for i in range(len(z_up)):
        # sampling points between z_+ and z_-
        z_close = np.linspace(z_low[i], z_up[i], 300)
        # calucalte rightmost integral
        num_integral1[i,:] = scipy.integrate.simps(integrand1, z_close)

    # calculate rightmost integral on denominator using same integrand as before
    # but over the full z_s range 
    denom_integral1 = scipy.integrate.simps(integrand1, z_s)

    # multiply solution to first integral by weights and integrate over z_ph
    # to find second integral
    num_integrand2 = weights * num_integral1
    num_integral2 = scipy.integrate.simps(num_integrand2, z_ph)

    denom_integrand2 = weights * denom_integral1
    denom_integral2 = scipy.integrate.simps(denom_integrand2, z_ph)

    # multiply solution to second integral by dNdz_l and integrate over z_l
    # to find full numerator and denominator
    num_integrand3 = dNdz_l * num_integral2
    num_integral3 = scipy.integrate.simps(num_integrand3, z_l)

    denom_integrand3 = dNdz_l * denom_integral2
    denom_integral3 = scipy.integrate.simps(denom_integrand3, z_l)

    # put numerator over denominator to find F
    F = num_integral3 / denom_integral3
    
    return F