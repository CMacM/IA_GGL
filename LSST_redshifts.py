import numpy as np
import scipy
import params_LSST_DESI as pa
import matplotlib.pyplot as plt
import pyccl as ccl

# Start by initialsising cosmology to carry through to halo_model.py and spurious_george.py
cosmo = ccl.Cosmology(Omega_c=pa.OmC, Omega_b=pa.OmB, h=(pa.HH0/100.), sigma8=pa.sigma8, n_s=pa.n_s_cosmo)

# set survey year
survey_year = 1

# LSST z-dist pamameters 
y1_z0_lens = 0.26
y1_alp_lens = 0.94
y1_z0_source = 0.13
y1_alp_source = 0.78

y10_z0_lens = 0.28
y10_alp_lens = 0.90
y10_z0_source = 0.11
y10_alp_source = 0.68

zl_max = 1.2
zs_max = 3.5
z_min = 0.2
z_vec_len = 300

def get_dndz_spec(gtype, year=survey_year):
    '''Construct spectroscopic distributions for sources and lenses'''
    
    if year == 1 and gtype == 'source':
        z_arr = np.linspace(z_min, zs_max, z_vec_len)
        dndz = z_arr**2 * np.exp(-(z_arr/y1_z0_source)**y1_alp_source)
        norm = scipy.integrate.simps(dndz, z_arr)
        dndz = dndz/norm
    elif year == 10 and gtype == 'source':
        z_arr = np.linspace(z_min, zs_max, z_vec_len)
        dndz = z_arr**2 * np.exp(-(z_arr/y10_z0_source)**y10_alp_source)
        norm = scipy.integrate.simps(dndz, z_arr)
        dndz = dndz/norm
    elif year == 1 and gtype == 'lens':
        z_arr = np.linspace(z_min, zl_max, z_vec_len)
        dndz = z_arr**2 * np.exp(-(z_arr/y1_z0_lens)**y1_alp_lens)
        norm = scipy.integrate.simps(dndz, z_arr)
        dndz = dndz/norm
    elif year == 10 and gtype == 'lens':
        z_arr = np.linspace(z_min, zl_min, z_vec_len)
        dndz = z_arr**2 * np.exp(-(z_arr/y10_z0_lens)**y10_alp_lens)
        norm = scipy.integrate.simps(dndz, z_arr)
        dndz = dndz/norm
    else:
        print('Not an LSST release year')
        z_arr, dndz, zeff = 0., 0., 0.
        
    zeff = np.average(z_arr, weights=get_weights()*np.ones([len(z_arr)]))
        
    return z_arr, dndz, zeff

def get_dndz_phot(gtype, year=survey_year, plot_fig='n', save_fig='n'):
    '''Convolves error probability distribution with spectroscopic distribution to approximate photometric distribution'''
    
    # get spectroscopic data
    z_s, dndz_s, zseff = get_dndz_spec(gtype, year)
    
    if gtype == 'lens':
        stretch = 0.2
        sig_z = 0.03*(1. + z_s)
    elif gtype == 'source':
        stretch = 0.5
        sig_z = 0.05*(1. + z_s)
    
    # set arbitrary photo-z points in extended redshift range
    z_ph = np.linspace(np.min(z_s), np.max(z_s)+stretch, 300)       
    
    # find probability of galaxy with true redshift z_s to be measured at redshift z_ph
    p_zs_zph = np.zeros([len(z_s),len(z_ph)])
    for zs in range(len(z_s)):
        p_zs_zph[zs,:] =  1. / (np.sqrt(2. * np.pi) * sig_z) * np.exp(-((z_ph - z_s[zs])**2) / (2. * sig_z**2))
        
    # integrate over z_s to get dN
    integrand1 = dndz_s * p_zs_zph
    integral1 = scipy.integrate.simps(integrand1, z_s, axis=0)
    dN = integral1
    
    # integrate dN over z_ph to get dz_ph 
    integrand2 = dN
    integral2 = scipy.integrate.simps(integrand2, z_ph)
    dz_ph = integral2

    dndz_ph = dN / dz_ph
    
    norm = scipy.integrate.simps(dndz_ph, z_ph)
    dndz_ph = dndz_ph / norm
    
    if plot_fig == 'y':
        plt.figure(figsize=[6,5])
        plt.title(gtype, fontsize=15)
        plt.plot(z_ph, dndz_ph)
        plt.plot(z_s, dndz_s)
        plt.xlim([0., 4.0])
        plt.xlabel('z', fontsize=15)
        plt.ylabel(r'$\frac{dn}{dz}$', fontsize=15)
        plt.legend([r'$z_{ph}$',r'$z_{s}$'], fontsize=16);
    
    if save_fig == 'y':
        plt.savefig('zdist_comparison_post_convolution.png', dpi=300)
        
    return z_ph, dndz_ph, p_zs_zph

def get_weights():
    '''Compute weights from LSST source ellipticity paramter forecasts. Current function is
    simple but can be extended in complexity'''
    
    weights = 1. / (pa.e_rms_mean**2 + pa.sig_e**2)
    
    return weights

def window(year=survey_year):
    """ Get window function, this is the window functions for LENSES x SOURCES. """
    """ Old note: Note I am just going to use the standard cosmological parameters here because it's a pain and it shouldn't matter too much. """
	
    z_l, dndz_l, zleff = get_dndz_spec('lens', year)
    
    # get source dndz but only out to max lens z
    if year == 10:
        z_s = np.linspace(z_min, zl_max, z_vec_len)
        dNdz_s = z_s**2 * np.exp(-(z_s/y10_z0_source)**y10_alp_source)
        norm = scipy.integrate.simps(dNdz_s, z_s)
        dndz_s = dNdz_s / norm
    elif year == 1:
        z_s = np.linspace(z_min, zl_max, z_vec_len)
        dNdz_s = z_s**2 * np.exp(-(z_s/y1_z0_source)**y1_alp_source)
        norm = scipy.integrate.simps(dNdz_s, z_s)
        dndz_s = dNdz_s / norm
    else:
        print('Not a survey year')
        
	
    chi = ccl.comoving_radial_distance(cosmo, 1./(1.+z_l)) * (pa.HH0_t / 100.) # CCL returns in Mpc but we want Mpc/h
    
    OmL = 1. - pa.OmC - pa.OmB - pa.OmR - pa.OmN
    
    dzdchi = pa.H0 * ( ( pa.OmC + pa.OmB )*(1+z_l)**3 + OmL + (pa.OmR+pa.OmN) * (1+z_l)**4 )**(0.5) 
        

    norm =  scipy.integrate.simps(dndz_l*dndz_s / chi**2 * dzdchi, z_l)
	
    win = dndz_l*dndz_s / chi**2 * dzdchi / norm
	
    return z_l, win
