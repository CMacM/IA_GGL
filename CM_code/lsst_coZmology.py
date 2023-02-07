import numpy as np
import scipy
import matplotlib.pyplot as plt
import pyccl as ccl
from more_itertools import locate

import DL_basis_code.params_LSST_DESI as pa

# cosmological parameters used in LSST SRD analysis
cosmo_SRD = ccl.Cosmology(Omega_c=pa.OmC, Omega_b=pa.OmB, h=(pa.HH0/100.), sigma8=pa.sigma8, n_s=pa.n_s_cosmo)

# cosmological parameters used in Nicola et al. HOD
cosmo_Nic = ccl.Cosmology(Omega_c=0.264, Omega_b=0.0493, h=0.6736, sigma8=0.8111, n_s=0.9649)

# cosmological parameters used in Zu & Mandelbaum HOD
cosmo_ZuMa = ccl.Cosmology(Omega_c=pa.OmC_s, Omega_b=pa.OmB_s, h=(pa.HH0/100.), sigma8=pa.sigma8_s, n_s=pa.n_s_s)

# set survey year
survey_year = 1

poolsize = 30

# LSST z-dist pamameters 
y1_z0_lens = 0.26
y1_alp_lens = 0.94
y1_z0_source = 0.13
y1_alp_source = 0.78

y10_z0_lens = 0.28
y10_alp_lens = 0.90
y10_z0_source = 0.11
y10_alp_source = 0.68

zs_max = 3.5
zs_min = 0.05
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
    for_Pi = False,
    normalise=True
):
    '''Convolves error probability distribution with spectroscopic 
    distribution to approximate photometric distribution'''
    
    # get spectroscopic data
    z_s, dndz_s, *_ = get_dndz_spec(gtype=gtype, zsmin=zsmin, zsmax=zsmax, 
                                       zlmin=zlmin, zlmax=zlmax, year=year, 
                                       z_vec_len=z_vec_len, normalise=False
                                      )
    
    if gtype == 'lens':
        stretch = 0.2
        sig_z = 0.03*(1. + z_s)
    elif gtype == 'source':
        stretch = 0.5
        sig_z = 0.05*(1. + z_s)
    
    # set arbitrary photo-z points in extended redshift range
    z_ph = np.linspace(zs_min, zs_max+stretch, z_vec_len)       
    
    # find probability of galaxy with true redshift z_s to be measured at redshift z_ph
    integrand1 = np.zeros([len(z_s),len(z_ph)])
    p_zs_zph = np.zeros([len(z_s),len(z_ph)])
    for zs in range(len(z_s)):
        for zph in range(len(z_ph)):
            p_zs_zph[zs,zph] =  (1. / (np.sqrt(2. * np.pi) * sig_z[zs])) * np.exp(-((z_ph[zph] - z_s[zs])**2) / (2. * sig_z[zs]**2))
            
        integrand1[zs,:] = dndz_s[zs] * p_zs_zph[zs,:]
    
    if for_Pi:
        # if this is to be used in the boost calculations, 
        # return before integrating.
        return integrand1
        
    # integrate over z_s to get dN
    integral1 = scipy.integrate.simps(integrand1, z_s, axis=0)
    dN = integral1
    
    if normalise==True:
        dz_ph = scipy.integrate.simps(dN, z_ph)
        dndz_ph = dN / dz_ph
    else:
        dndz_ph = dN
        
    return z_ph, dndz_ph, p_zs_zph

def get_weights(sig_y=pa.e_rms_mean, sig_e=pa.sig_e):
    '''Compute weights from LSST source ellipticity paramter forecasts. 
    Current function issimple but can be extended in complexity'''
    
    weights = 1. / (sig_y**2 + sig_e**2)
    
    return weights

def window(
    zsmin=zs_min, 
    zlmax=zl_max,
    year=survey_year,
    z_vec_len=z_vec_len
):
    """ Get window function, this is the window functions for LENSES x SOURCES. """

    z_l, dndz_l, zleff = get_dndz_spec(gtype='lens',zlmin=zsmin,zlmax=zlmax,year=year,z_vec_len=z_vec_len)
    
    # get source dndz but only out to max lens z
    if year == 10:
        z_s = np.linspace(zsmin, zlmax, z_vec_len)
        dNdz_s = z_s**2 * np.exp(-(z_s/y10_z0_source)**y10_alp_source)
        norm = scipy.integrate.simps(dNdz_s, z_s)
        dndz_s = dNdz_s / norm
    elif year == 1:
        z_s = np.linspace(zsmin, zlmax, z_vec_len)
        dNdz_s = z_s**2 * np.exp(-(z_s/y1_z0_source)**y1_alp_source)
        norm = scipy.integrate.simps(dNdz_s, z_s)
        dndz_s = dNdz_s / norm
    else:
        print('Not a survey year')
        
    chi = ccl.comoving_radial_distance(cosmo_SRD, 1./(1.+z_l)) * (pa.HH0_t / 100.) # CCL returns in Mpc but we want Mpc/h
    
    OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
    
    dzdchi = pa.H0 * ( ( pa.OmC + pa.OmB )*(1+z_l)**3 + OmL + (pa.OmR+pa.OmN) * (1+z_l)**4 )**(0.5) 

    norm =  scipy.integrate.simps(dndz_l*dndz_s / chi**2 * dzdchi, z_l)

    win = dndz_l*dndz_s / chi**2 * dzdchi / norm

    return z_l, win
