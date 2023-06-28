import numpy as np
import scipy
import matplotlib.pyplot as plt
import pyccl as ccl
import math
from more_itertools import locate

import DL_basis_code.params_LSST_DESI as pa

# cosmological parameters used in LSST SRD analysis
cosmo_SRD = ccl.Cosmology(Omega_c=pa.OmC, Omega_b=pa.OmB, h=(pa.HH0/100.), sigma8=pa.sigma8, n_s=pa.n_s_cosmo)

# cosmological parameters used in Nicola et al. HOD
cosmo_Nic = ccl.Cosmology(Omega_c=0.264, Omega_b=0.0493, h=0.6736, sigma8=0.8111, n_s=0.9649)

# cosmological parameters used in Zu & Mandelbaum HOD
cosmo_ZuMa = ccl.Cosmology(Omega_c=pa.OmC_s, Omega_b=pa.OmB_s, h=(pa.HH0_s/100.), sigma8=pa.sigma8_s, n_s=pa.n_s_s)

# set survey year, default is 1
survey_year = 1

# set poolsize for parallelised functions
poolsize = 30

# LSST z-dist pamameters 
y1_z0_lens = 0.26
y1_alp_lens = 0.94

y10_z0_lens = 0.28
y10_alp_lens = 0.90

# r band size cut parameters
y1_z0_source = 0.11
y1_alp_source = 0.71

y10_z0_source = 0.06
y10_alp_source = 0.57

# choose default tomographic binning
zs_min = 0.05
zs_max = 3.5

zl_min = 0.2
zl_max = 1.2

z_vec_len = 300

def get_dndz_spec(
    gtype, 
    zsmin=zs_min, 
    zsmax=zs_max, 
    zlmin=zl_min, 
    zlmax=zl_max,
    year=survey_year, 
    z_vec_len=z_vec_len, 
    normalise=True
):
    '''Construct spectroscopic distributions for sources and lenses'''
    
    if year == 1 and gtype == 'source':
        z_arr = np.linspace(zsmin, zsmax, z_vec_len)
        dndz = z_arr**2 * np.exp(-(z_arr/y1_z0_source)**y1_alp_source)
        if normalise==True:
            norm = scipy.integrate.simps(dndz, z_arr)
            dndz = dndz/norm
    elif year == 10 and gtype == 'source':
        z_arr = np.linspace(zsmin, zsmax, z_vec_len)
        dndz = z_arr**2 * np.exp(-(z_arr/y10_z0_source)**y10_alp_source)
        if normalise==True:
            norm = scipy.integrate.simps(dndz, z_arr)
            dndz = dndz/norm
    elif year == 1 and gtype == 'lens':
        z_arr = np.linspace(zlmin, zlmax, z_vec_len)
        dndz = z_arr**2 * np.exp(-(z_arr/y1_z0_lens)**y1_alp_lens)
        if normalise==True:
            norm = scipy.integrate.simps(dndz, z_arr)
            dndz = dndz/norm
    elif year == 10 and gtype == 'lens':
        z_arr = np.linspace(zlmin, zlmax, z_vec_len)
        dndz = z_arr**2 * np.exp(-(z_arr/y10_z0_lens)**y10_alp_lens)
        if normalise==True:
            norm = scipy.integrate.simps(dndz, z_arr)
            dndz = dndz/norm
    else:
        print('Not an LSST release year')
        z_arr, dndz, zeff = 0., 0., 0.
        
    zeff = np.average(z_arr, weights=get_weights()*np.ones([len(z_arr)]))
    return z_arr, dndz, zeff

def get_dndz_phot(
    gtype, 
    zsmin=zs_min, 
    zsmax=zs_max, 
    zlmin=zl_min, 
    zlmax=zl_max,
    year=survey_year, 
    z_vec_len=z_vec_len,
    for_Pi=False,
    z_Pi=None,
    normalise=True,
    z_shift=None
):
    '''Convolves error probability distribution with spectroscopic 
    distribution to approximate photometric distribution'''
    
    # get spectroscopic data
    # this can be a lower res vector
    z_s, dndz_s, *_ = get_dndz_spec(gtype=gtype, zsmin=zsmin, zsmax=zsmax, 
                                       zlmin=zlmin, zlmax=zlmax, year=year, 
                                       z_vec_len=z_vec_len, normalise=False
                                      )
    
    if z_shift is not None:
        z_s = z_shift[0]
        dndz_s = z_shift[1]
    
    if year == 1 and z_Pi is not None:
        z_s = z_Pi
        dndz_s = z_s**2 * np.exp(-(z_s/y1_z0_source)**y1_alp_source)
    elif year == 10 and z_Pi is not None:
        z_s = z_Pi
        dndz_s = z_s**2 * np.exp(-(z_s/y10_z0_source)**y10_alp_source)

    if gtype == 'lens':
        stretch = 0.2
        sig_z = 0.03*(1. + z_s)
    elif gtype == 'source':
        stretch = 0.5
        sig_z = 0.05*(1. + z_s)
        
#     chi_plus = ccl.comoving_radial_distance(cosmo_SRD,
#                     1./(1. + np.max(z_s))) * (pa.HH0/100.) + 500
#     a_plus = ccl.scale_factor_of_chi(cosmo_SRD, chi_plus / (pa.HH0/100.))
#     zph_plus = (1. / a_plus) - 1. 
    
#     chi_minus = ccl.comoving_radial_distance(cosmo_SRD,
#                     1./(1. + np.min(z_s))) * (pa.HH0/100.) - 500
#     if chi_minus < 0.:
#         zph_minus = 0.
#     else:
#         a_minus = ccl.scale_factor_of_chi(cosmo_SRD, chi_minus / (pa.HH0/100.))
#         zph_minus = (1. / a_minus) - 1. 
    
#     # set arbitrary photo-z points in extended redshift range
#     #z_ph = np.linspace(zph_minus, zph_plus, z_vec_len)
    
    z_ph = np.linspace(0., 4., z_vec_len)
                           
    # find probability of galaxy with true redshift z_s to be measured at redshift z_ph
    integrand1 = np.zeros([len(z_s),len(z_ph)])
    p_zs_zph = np.zeros([len(z_s),len(z_ph)])
    for zs in range(len(z_s)):
        
        p_zs_zph[zs,:] =  (1. / (np.sqrt(2. * np.pi) * sig_z[zs])) * np.exp(-((z_ph - z_s[zs])**2) / (2. * sig_z[zs]**2))

    integrand1 = p_zs_zph * dndz_s[:,None]   
    
    if for_Pi:
        # if this is to be used in the boost calculations, 
        # return before integrating.
        return z_ph, dndz_s, p_zs_zph
        
    # integrate over z_s to get dN
    integral1 = scipy.integrate.simps(integrand1, z_s, axis=0)
    dN = integral1
    
    if normalise==True:
        dz_ph = scipy.integrate.simps(dN, z_ph)
        dndz_ph = dN / dz_ph
    else:
        dndz_ph = dN
        
    return z_ph, dndz_ph, p_zs_zph

def get_weights(sig_y=pa.e_rms_mean, sig_e=pa.sig_e, simple=True, zs=None, zl=None):
    '''Compute weights from LSST source ellipticity paramter forecasts. 
    Current function issimple but can be extended in complexity'''
    
    if simple:
        w = 1. / (sig_y**2 + sig_e**2)
    else:
        print('More complex weighting schemes are not yet implemented')
        
    return w

def window(
    zlmin=zl_min,
    zlmax=zl_max,
    zsmin=zs_min,
    zsmax=zs_max,
    year=survey_year,
    z_vec_len=z_vec_len
):
    """ Get window function, this is the window functions for LENSES x SOURCES. """
    
    # get lens redshift data
    z_l, dndz_l, zleff = get_dndz_spec(gtype='lens',
                                       zlmin=zlmin,
                                       zlmax=zlmax,
                                       year=year,
                                       z_vec_len=z_vec_len)
    
    # get source redshifts data in lens redshift range
    z_s, dndz_s, zseff = get_dndz_phot(gtype='source',
                                       zsmin=zlmin,
                                       zsmax=zlmax,
                                       year=year,
                                       z_vec_len=z_vec_len)
        
    chi = ccl.comoving_radial_distance(cosmo_SRD, 1./(1.+z_l)) * (pa.HH0_t / 100.) # CCL returns in Mpc but we want Mpc/h
    
    OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
    
    dzdchi = pa.H0 * ( ( pa.OmC + pa.OmB )*(1+z_l)**3 + OmL + (pa.OmR+pa.OmN) * (1+z_l)**4 )**(0.5) 

    norm =  scipy.integrate.simps(dndz_l*dndz_s / chi**2 * dzdchi, z_l)

    win = dndz_l*dndz_s / chi**2 * dzdchi / norm

    return z_l, win

def get_dndz_weighted(
    gtype,
    z,
    dndz,
    z0=None,
    year=survey_year, 
    z_vec_len=z_vec_len,
    normalise=True,
):
    '''Convolves z dist with a shifted distribution
    distribution to approximate a weighted redshift distribution''' 
    
    if year==1:
        alp = y1_alp_source
    elif year==10:
        alp = y10_alp_source
    
    # create distribution with shifted mean and convolve it
    dndz_shift = z**2 * np.exp(-(z/z0)**alp)
        
    # integrate over z_s to get dN
    
    if normalise==True:
        dndz_shift = dndz_shift / scipy.integrate.simps(dndz_shift, z)

    return z, dndz_shift

