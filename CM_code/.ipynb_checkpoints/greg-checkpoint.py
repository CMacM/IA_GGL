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
import multiprocessing as mp
import time 
#import main.py as main

import halo_model as halo

while 1==1:
    poolsize = int(input('spurious george: Please define poolsize to be used for parallelised functions'))
    if isinstance(poolsize, int) == True:
        print('poolsize set to %d'%poolsize)
        break
    else:
        continue

# set up everything in terms of theta
theta_min = 2.5
theta_max = 250
N_bins = 10
interp_len = 1000

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

# define theta vector to convert to r_p and use in final plot
theta_edges = setup.setup_rp_bins(theta_min, theta_max, N_bins)
theta_cents = setup.rp_bins_mid(theta_edges)

'''OLD VERSION WITH SLIGHTLY UPDATED REDSHIFT DISTS AND NORMALISATIONS TO CHECK WHAT WENT WRONG IN GEORGE'''

# simply pulls spec-z distributions and z values from LSST forecast data release
# distributions should already be normalised
def get_dndz_spec(gtype, year):
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
        z_arr, dndz = 0.
        
    zeff = np.average(z_arr, weights=get_weights()*np.ones([len(z_arr)]))
        
    return z_arr, dndz, zeff

def get_dNdz_phot(gtype, year, plot_fig='n', save_fig='n'):
    '''Convolves error probability distribution with spectroscopic distribution to approximate photometric distribution'''
    
    # get spectroscopic data
    z_s, dNdz_s, zseff = get_dndz_spec(gtype, year)
    
    # set arbitrary photo-z points in extended redshift range
    z_ph = np.linspace(np.min(z_s), np.max(z_s)+0.5, 300)

    if gtype == 'source':
        sig_z = 0.05*(1. + z_s)
    elif gtype == 'lens':
        sig_z = 0.03*(1. + z_s)
    
    # find probability of source with true redshift z_s to be measured at redshift z_ph
    p_zs_zph = np.zeros([len(z_s),len(z_ph)])
    for zs in range(len(z_s)):
        p_zs_zph[zs,:] =  1. / (np.sqrt(2. * np.pi) * sig_z) * np.exp(-((z_ph - z_s[zs])**2) / (2. * sig_z**2))
        
    # integrate over z_s to get dN
    integrand1 = dNdz_s * p_zs_zph
    integral1 = scipy.integrate.simps(integrand1, z_s, axis=0)
    dN = integral1
    
    # integrate dN over z_ph to get dz_ph 
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

def get_IA_zlims(year):
    
    #get LSST forecats spec-z data
    z_l, dNdz_l, zleff = get_dndz_spec(gtype='lens', year=year)
    zleff = np.average(z_l, weights=get_weights()*np.ones([len(z_l)]))
    
    # set some arbitrary projected separations, r_p from theta
    r_p = setup.arcmin_to_rp(theta_cents, zleff, halo.cosmo) * (pa.HH0/100)

    #convert source and lens spec-z to com-ra-dist
    chi_lens = ccl.comoving_radial_distance(halo.cosmo, 1./(1. + z_l)) * (pa.HH0/100.) # convert to Mpc/h

    chi_up = np.zeros([len(z_l), len(r_p)]) 
    chi_low = np.zeros([len(z_l), len(r_p)])
    z_up = np.zeros([len(z_l), len(r_p)])
    z_low = np.zeros([len(z_l), len(r_p)])
    #find upper and lower IA lims in com-ra-dist
    for zl in range(len(z_l)):
        # implement new definition for maximum L.O.S separation
        chi_up[zl,:] = [chi_lens[zl] + rp if rp > 2. else chi_lens[zl] + 2. for rp in r_p]
        chi_low[zl,:] = [chi_lens[zl] - rp if rp > 2. else chi_lens[zl] - 2. for rp in r_p]
        
        # remove negative distances from resulting lower lims
        chi_low[zl,:] = [0. if x < 0. else x for x in chi_low[zl,:]]

        #convert back to redshift
        z_up[zl,:] = (1./ ccl.scale_factor_of_chi(halo.cosmo, chi_up[zl] / (pa.HH0/100.)) ) - 1.
        z_low[zl,:] = (1./ ccl.scale_factor_of_chi(halo.cosmo, chi_low[zl] / (pa.HH0/100.)) ) - 1.
    
    return z_up, z_low

def get_weights():
    
    weights = 1. / (pa.e_rms_mean**2 + pa.sig_e**2)
    
    return weights

def get_F(year):
    '''Calculated F(theta) from LSST forecast spectroscopic distributions and estimated
    phot-z distribtuion'''
    
    # get weights
    weights = get_weights()

    # get spec-z for sources and lenses
    z_s, dNdz_s, zseff = get_dNdz_spec(gtype='source', year=year)
    z_l, dNdz_l, zleff = get_dNdz_spec(gtype='lens', year=year)

    # convolve distributions to estimate dNdz_ph, z_ph is
    # zs_min <= z_s <= zs_max + 0.5
    z_ph, dNdz_ph, p_zs_zph = get_dNdz_phot(gtype='source', year=year)

    z_up, z_low = get_IA_zlims(year=year)

    # this gets a little confusing...
    F_theta = np.zeros([vec_len])
    for rp in range(vec_len):
        # calculate integrand for rightmost numerator integral
        integrand1 = dNdz_s * p_zs_zph 

        # loop over z_+ and z_- for different z_l and integrate between them
        num_integral1 = np.zeros(np.shape(p_zs_zph))
        for zl in range(len(z_l)):
            # sampling points between z_+ and z_-
            z_close = np.linspace(z_low[zl,rp], z_up[zl,rp], len(z_l))
            # calucalte rightmost integral
            num_integral1[zl,:] = scipy.integrate.simps(integrand1, z_close)

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
        F_theta[rp] = num_integral3 / denom_integral3
    
    return F_theta

def get_xi_gg(year, maxPi, minPi):

    # get spec-z for lenses
    z_l, dNdz_l, zleff = get_dndz_spec(gtype='lens', year=year)

    # convert redshifts for lenses and sources to scale factors
    La_arr = 1. / (1. + z_l)
    
    r_p = setup.arcmin_to_rp(theta_cents, zleff, halo.cosmo) * (pa.HH0 / 100.) #Mpc/h
    
    rmax = np.sqrt(np.max(r_p)**2 + (maxPi+1)**2)
    rmin = np.sqrt(np.min(r_p)**2 - minPi**2)

    # set some arbitrary 3D separations r to compute xi_gg
    r = np.logspace(np.log10(rmin), np.log10(rmax), interp_len)

    # wavenumbers and scale factors for all-space power spectrum
    k_arr = np.geomspace(1E-4, 5E2, 3*interp_len)

    # must flip array because ccl wants scale factors ascending (z_0 == a_1)
    pk_gg2D = halo.get_Pk2D('gg', k_arr=k_arr, a_arr=np.flip(La_arr))
    
    # use 2D power spectrum and lens-source 3D separations to get 3D correlation function xi_ls(r,z_l)
    xi_gg = np.zeros([len(r), len(z_l)])
    for zl in range(len(z_l)):
        xi_gg[:,zl] = ccl.correlations.correlation_3d(halo.cosmo, np.flip(La_arr[zl]), 
                                           r / (pa.HH0 / 100.), pk_gg2D)
    
    return xi_gg, r_p, r

def get_xi_ls(year):

    # get spec-z for sources and lenses
    z_s, dNdz_s, zseff = get_dndz_spec(gtype='source', year=year)
    z_l, dndz_l, zleff = get_dndz_spec(gtype='lens', year=year)

    # convert redshifts for lenses and sources to scale factors
    La_arr = 1. / (1. + z_l)
    Sa_arr = 1. / (1. + z_s)

    # comoving radial distance from observer to sources and lenses
    chi_lens = ccl.comoving_radial_distance(halo.cosmo, La_arr) * (pa.HH0 / 100.)
    chi_source = ccl.comoving_radial_distance(halo.cosmo, Sa_arr) * (pa.HH0 / 100.)

    # need to flip array so highest z comes first for iteration routine below and because ccl wants scale factors backwards, 1 -> 0
    chi_flipped = np.flip(chi_lens)

    # L.o.S pair separation array
    minPiPos = 10**(-3) 
    max_int = 200. # We integrate the numerator out to 200 Mpc/h away from the lenses because the correlation is zero outside of this.
    maxPiPos = max_int
    Pi_pos = np.logspace(np.log10(minPiPos), np.log10(maxPiPos), interp_len)

    # Pi can be positive or negative, so now flip this and include the negative values, but only down to z=0
    # And avoid including multiple of the same values - this messes up some integration routines.
    Pi = [0]*len(z_l)
    try:
        for zi in range(len(z_l)):
            Pi_pos_vec= list(Pi_pos)[1:]
            Pi_pos_vec.reverse()
            index_cut = next(j[0] for j in enumerate(Pi_pos_vec) if j[1]<=(chi_flipped[zi]-np.min(chi_source)))
            Pi[zi] = np.append(-np.asarray(Pi_pos_vec[index_cut:]), np.append([0],Pi_pos))  
    except StopIteration:
        Pi[-1] = Pi_pos
    
    # call to other function to calculate xi_gg
    xi_gg, r_p, r = get_xi_gg(year, maxPiPos, minPiPos)
    
    # define function used to calculate xi_ls(r,Pi) at fixed zl
    def calculate_xi_ls(zl, r, r_p, Pi, xi_gg):
        # xi_gg is a function of r, so we interpolate it across r to later find x_ls as a function of rp and pi
        xi_interp_r = scipy.interpolate.interp1d(r, xi_gg[:, zl])
        # preallocate array for xi(rp,pi(zl))
        xi_ofPi = np.zeros((len(r_p), len(Pi[zl])))
        for ri in range(len(r_p)):
            for pi in range(len(Pi[zl])):
                rp_pi_zl = np.sqrt((r_p[ri]**2 + Pi[zl][pi]**2))
                xi_ofPi[ri, pi] = xi_interp_r(rp_pi_zl)

        return xi_ofPi

    # wrap function inside another function so it can be passsed to multiprocessing with a single set
    # of iterables
    global wrap_for_mp
    def wrap_for_mp(iterable):
        return calculate_xi_ls(zl=iterable, r=r, r_p=r_p, Pi=Pi, xi_gg=xi_gg), iterable

    # how many iterations do you want it done in?
    iteration_number = 10

    # set size of pool used, easier to define this way so we optimise number of processes 
    # allocated
    poolsize = int(len(z_l) / iteration_number)

    # preallocate list for outputs
    xi_ls = [0] * len(z_l)

    # set first set of indices to be passed to multiprocessing and computed 
    iterables = list(range(len(z_l)))

    # open multiprocess pool in with statement so it closes after it's complete
    start = time.time()
    with mp.Pool(poolsize) as p:
        # output values and process ID (imap should ensure correct order)
        xi_ls, order = zip(*p.imap(wrap_for_mp, iterables))

    # run quick test to ensure order of outputs was preserved
    check_order = list(order)
    if check_order == iterables:
        print('Order preserved for zl indices %d-%d'%(iterables[0],iterables[-1]))
    else:
        print('Order scrambled, data corrupted, exiting...')
        exit()
    
    end = time.time()
    print('Runtime: %g minutes'%((end-start)/60))
    
    com_Pi = [0]*len(z_l)
    z_Pi = [0]*len(z_l)
    # get comoving distance values associated to Pi
    for zl in range(len(z_l)):
        com_Pi[zl] = chi_flipped[zl] + Pi[zl]
        # CCL wants distance in Mpc but we are working in Mpc/h
        z_Pi[zl] = (1./ccl.scale_factor_of_chi(halo.cosmo, com_Pi[zl] / (pa.HH0_t/100.))) - 1. 

    del com_Pi

    return xi_ls, r_p, Pi, z_Pi, z_l, dndz_l

def get_boosts(year):
    
    xi_ls, r_p, Pi, z_Pi, z_l, dndz_l = get_xi_ls(year)
    
    # get weights
    weights = get_weights()
    
    if year == 1:
        z0 = y1_z0_source
        alpha = y1_alp_source
    elif year == 10:
        z0 = y10_z0_source
        alpha = y10_alp_source
    else: 
        print('Not a current survey year')
        exit()

    z_s, dndz_s, zseff = get_dndz_spec('source', year)
    
    # start by constructing dndz_Pi using SRD parameterisation
    z_Pi_norm = [0] * len(z_l)
    dndz_Pi = [0] * len(z_l)
    dndz_Pi_norm = [0] * len(z_l)
    zph_norm = [0] * len(z_l)

    for zi in range(len(z_l)):
        # how do I normalise this?
        dndz_Pi[zi] = z_Pi[zi]**2 * np.exp(-(z_Pi[zi]/z0) ** alpha)
            
        # need to normalise over the full spec-z range
        z_Pi_norm[zi] = np.linspace(np.min(z_s), np.max(z_s), len(z_Pi[zi]))
        dndz_Pi_norm[zi] = z_Pi_norm[zi]**2 * np.exp(-(z_Pi_norm[zi]/z0) ** alpha)
        
        # set full range of photo-z to normalise over
        zph_norm[zi] = np.linspace(np.min(z_s), np.max(z_s)+0.5, len(z_Pi[zi]))
    
    # next get error dist p_zPi_zph
    p_zPi_zph = [0] * len(z_l)
    p_zPi_zph_norm = [0] * len(z_l)

    # first loop over z_Pi values for each lens redshift
    for zl in range(len(z_l)):
        
        # define photo-z shift value for each zl
        chi_plus = ccl.comoving_radial_distance(halo.cosmo, 
                                            1./(1. + np.max(z_Pi[zl]))) * (pa.HH0/100.) + np.max(Pi[zl])
        a_plus = ccl.scale_factor_of_chi(halo.cosmo, chi_plus / (pa.HH0/100.))
        zph_plus = (1. / a_plus) - 1.
        # set arbitrary photo-z values in range of z_Pi boosted by 200Mpc in z                                 
        zPi_ph = np.linspace(np.min(z_Pi[zl]), zph_plus, len(z_Pi[zl]))
        
        # variance for each z_Pi
        sig_z = 0.05*(1. + z_Pi[zl])
        sig_z_norm = 0.05*(1. + z_Pi_norm[zl])
        
        p = np.zeros([len(zPi_ph), len(z_Pi[zl])])
        p_norm = np.zeros([len(zPi_ph), len(z_Pi[zl])])
        
        # loop over specific z_Pi values to find error dist for each z_Pi, also include normalisation over full redshift range
        for zi in range(len(z_Pi[zl])):
            # create 2D array of z_ph probability at each z_Pi
            p[zi,:] =  1. / (np.sqrt(2. * np.pi) * sig_z) * np.exp(-((zPi_ph - z_Pi[zl][zi])**2) / (2. * sig_z**2))
            p_norm[zi,:] =  1. / (np.sqrt(2. * np.pi) * sig_z_norm) * np.exp(-((zph_norm[zl] - z_Pi_norm[zl][zi])**2) / (2. * sig_z_norm**2))
            
        # save p(z_Pi,z_ph) at each lens redshift
        p_zPi_zph[zl] = p
        # need to also get error pdf for full zph to normalise over
        p_zPi_zph_norm[zl] = p_norm
        
    def integrate_zPi(zl):
        # preallocate arrays to store data
        igl_zPi = np.zeros([len(r_p), len(z_Pi[zl])])
        num_igd_zPi = np.zeros([len(z_Pi[zl]), len(z_Pi[zl])])
        denom_igd_zPi = np.zeros([len(z_Pi[zl]), len(z_Pi[zl])])
        for ri in range(len(r_p)):
            # rows are rp, columns represent different Pi values
            num_igd_zPi = dndz_Pi[zl] * xi_ls[zl][ri,:] * p_zPi_zph[zl]
            denom_igd_zPi = dndz_Pi_norm[zl] * p_zPi_zph_norm[zl]
            # for P(zpi,zph), we want to integrate over the z_Pi values, these are rows so axis=0
            igl_zPi[ri,:] = scipy.integrate.simps(num_igd_zPi, z_Pi[zl], axis=0) / scipy.integrate.simps(denom_igd_zPi, z_Pi_norm[zl])

        # this leaves us with an array that is [len(rp) x len(zph)]
        return igl_zPi
    
    global integrate_zph
    def integrate_zph(zl):
        # now we want to integrate over zph, so we need array to store integrals for different rp
        igl_zph = np.zeros([len(r_p)])
        # arbitrarily define photometric redshifts slightly extended in range from zPi
        chi_plus = ccl.comoving_radial_distance(halo.cosmo, 
                                            1./(1. + np.max(z_Pi[zl]))) * (pa.HH0/100.) + np.max(Pi[zl])
        a_plus = ccl.scale_factor_of_chi(halo.cosmo, chi_plus / (pa.HH0/100.))
        zph_plus = (1. / a_plus) - 1. 
        zPi_ph = np.linspace(np.min(z_Pi[zl]), zph_plus, len(z_Pi[zl]))
        # call to previous function to get first integral
        igd_zph = weights * integrate_zPi(zl)
        # integrate over zph for a certain zl, zph are the columns so specify axis=1
        igl_zph = scipy.integrate.simps(igd_zph, zPi_ph, axis=1) / scipy.integrate.simps(weights * np.ones([len(zPi_ph)]), zph_norm[zl])

        # return len(rp) array representing [B(rp) - 1] at zl 
        return igl_zph, zl
    
    # parallelised integration routine
    iterables = list(range(len(z_l)))

    poolsize = 30

    # allocate array to store [B(rp) - 1] for each zl
    igl_zph = np.zeros([len(z_l), len(r_p)])

    start = time.time()
    with mp.Pool(poolsize) as p:
        # output values and process ID (imap should ensure correct order)
        igl_zph, order = zip(*p.map(integrate_zph, iterables))

    # run quick test to ensure order of outputs was preserved
    check_order = list(order)
    if check_order == iterables:
        print('Order preserved for zl indices %d-%d'%(iterables[0],iterables[-1]))
    else:
        print('Order scrambled, data corrupted, exiting...')
        exit()

    # replace NaN with 0. NaN is produced due to divide by zero error 
    #resulting from Pi[-1][:] = 0
    if np.isnan(igl_zph[-1], out=np.empty(len(igl_zph[-1]))).all() == True:
        igl_zph[-1][:] = np.zeros([len(igl_zph[-1])])

    # All [B(rp) - 1]_zl have been collected into a [len(z_l) x len(r_p)] array 
    # We must flip along the rows as previous integrals were collected with zl descending, 
    # but we integrate over an ascending sample
    igd_zl = np.flip(igl_zph, axis=0) * dndz_l[:,None]
    igl_zl = scipy.integrate.simps(igd_zl, z_l, axis=0) / scipy.integrate.simps(dndz_l, z_l)

    np.savez('Boost_data_year%d'%year, B_min1 = igl_zl, B = igl_zl + 1., rp = r_p, theta = theta_cents)

    end = time.time()
    print('Runtime = %g'%(end-start))
    
    return igl_zl, theta_cents


# Worry about these later
# ------------------------------------------------------------------------------------------------
def get_LSST_aps(year, ell, k_arr, a_arr, b=1.0, save_fig='n'):
    
    if year == 1:
        lens_z, lens_dNdz = get_dNdz_spec(gtype='lens', year=1)
        source_z, source_dNdz = get_dNdz_spec(gtype='source', year=1)
    elif year == 10:
        lens_z, lens_dNdz = get_dNdz_spec(gtype='lens', year=10)
        source_z, source_dNdz = get_dNdz_spec(gtype='source', year=10)
      
    # init tracers for lenses and sources
    lensTracer = ccl.NumberCountsTracer(halo.cosmo, has_rsd=False, dndz=(lens_z, lens_dNdz), 
                                    bias=(lens_z, b*np.ones(len(lens_z))))
    sourceTracer = ccl.WeakLensingTracer(halo.cosmo, (source_z, source_dNdz))
    
    # create 2D power spectrum objects for galaxy-shear correlation
    pk_gMf = halo.get_Pk2D('gm', k_arr, a_arr)
    
    # need to make sure pk_gMf is passed in!
    aps = ccl.angular_cl(halo.cosmo, lensTracer, sourceTracer, ell, p_of_k_a=pk_gMf)
    
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
    
    gammat = ccl.correlation(halo.cosmo, ell, aps, theta_arr, type='NG')
    
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