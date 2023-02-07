import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import os.path
import pyccl as ccl
import multiprocessing as mp
import time
from importlib import reload

import DL_basis_code.params_LSST_DESI as pa
import DL_basis_code.shared_functions_setup as setup

import CM_code.halo_model as halo
reload(halo)

import CM_code.lsst_coZmology as zed
reload(zed)

data_dir = '/home/b7009348/WGL_project/LSST_forecast_code/generated_data/'

# set mp poolsize for parallelised functions
poolsize = zed.poolsize

# set up everything in terms of theta
theta_min = 0.2
theta_max = 25
N_bins = 10
interp_len = 1000

survey_year = zed.survey_year

y1_lens_b = 1.05
y10_lens_b = 0.95

# define theta vector to convert to r_p and use in final plot
theta_edges = setup.setup_rp_bins(theta_min, theta_max, N_bins)
theta_cents = setup.rp_bins_mid(theta_edges)

'''Objective is to use script as a basis for computing the effects of spurious signals'''
######## FUNCTIONS BELOW ARE USED FOR LSST DATA RETRIEVAL, BOOST and F calculations ##########

def get_IA_zlims(year=survey_year):
    
    #get LSST forecats photo-z lens data
    z_l, dndzl_ph, p_zs_zlph = zed.get_dndz_phot(gtype='lens', year=year)
    zleff = np.average(z_l, weights=zed.get_weights()*np.ones([len(z_l)]))
    
    # set some arbitrary projected separations, r_p from theta
    r_p = setup.arcmin_to_rp(theta_cents, zleff, zed.cosmo_SRD) * (pa.HH0/100)

    #convert source and lens spec-z to com-ra-dist
    chi_lens = ccl.comoving_radial_distance(zed.cosmo_SRD, 1./(1. + z_l)) * (pa.HH0/100.) # convert to Mpc/h

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
        z_up[zl,:] = (1./ ccl.scale_factor_of_chi(zed.cosmo_SRD, chi_up[zl] / (pa.HH0/100.)) ) - 1.
        z_low[zl,:] = (1./ ccl.scale_factor_of_chi(zed.cosmo_SRD, chi_low[zl] / (pa.HH0/100.)) ) - 1.
    
    return z_up, z_low

def get_F(theta_dependence=True, year=survey_year):
    '''Calculated F(theta) from LSST forecast photo-z distributions'''
    
    # get weights
    weights = zed.get_weights()

    # get spec-z for sources
    z_s, dndz_s, zseff = zed.get_dndz_spec(gtype='source', year=year)

    # convolve distributions to estimate dndz_ph, z_ph is
    # zs_min <= z_s <= zs_max + 0.5
    zs_ph, dndzs_ph, p_zs_zsph = zed.get_dndz_phot(gtype='source', year=year)

    # use photometric lens redshifts
    z_l, dndz_l, p_zl_zlph = zed.get_dndz_phot(gtype='lens', year=year)
    
    
    if theta_dependence==True:

        # get limits for intrisnic alingment from lens photo-z
        z_up, z_low = get_IA_zlims(year=year)

        # this gets a little confusing...
        # F of theta because IA lims were defined in rp range based upon theta
        F = np.zeros([N_bins])
        for rp in range(N_bins):
            # calculate integrand for rightmost numerator integral
            integrand1 = dndz_s * p_zs_zsph 

            # loop over z_+ and z_- for different z_l and integrate between them
            num_integral1 = np.zeros(np.shape(p_zs_zsph))
            for zl in range(len(z_l)):
                # sampling points between z_+ and z_-
                z_close = np.linspace(z_low[zl,rp], z_up[zl,rp], len(z_l))
                # calucalte rightmost integral
                num_integral1[zl,:] = scipy.integrate.simps(integrand1, z_close)

            # calculate rightmost integral on denominator using same integrand as before
            # but over the full z_s range 
            integral1 = num_integral1 / scipy.integrate.simps(integrand1, z_s)

            # multiply solution to first integral by weights and integrate over z_ph
            # to find second integral
            integrand2 = weights * integral1
            integral2 = scipy.integrate.simps(integrand2, zs_ph) / scipy.integrate.simps(weights * np.ones(len(zs_ph)), zs_ph)

            # multiply solution to second integral by dndz_l and integrate over z_l
            # to find full numerator and denominator
            integrand3 = dndz_l * integral2
            integral3 = scipy.integrate.simps(integrand3, z_l) / scipy.integrate.simps(dndz_l, z_l)

            # put numerator over denominator to find F
            F[rp] = integral3
            
    else:

        #convert source and lens spec-z to com-ra-dist
        chi_lens = ccl.comoving_radial_distance(zed.cosmo_SRD, 1./(1. + z_l))
        chi_source = ccl.comoving_radial_distance(zed.cosmo_SRD, 1./(1. + z_s))

        #find upper and lower IA lims in com-ra-dist
        chi_up = chi_lens + 100.
        chi_low = chi_lens - 100.

        # remove negative distances from result lower lims
        chi_low = [0. if x < 0. else x for x in chi_low]

        #get scale factors for com-ra-dist lims
        a_up = ccl.scale_factor_of_chi(zed.cosmo_SRD, chi_up)
        a_low = ccl.scale_factor_of_chi(zed.cosmo_SRD, chi_low)

        #convert scale factors to find lims in redshift space
        z_up = (1./a_up) - 1.
        z_low = (1./a_low) - 1.

        # this gets a little confusing... 

        # calculate integrand for rightmost numerator integral
        integrand1 = dndz_s * p_zs_zsph

        # loop over z_+ and z_- for different z_l and integrate between them
        num_integral1 = np.zeros(np.shape(p_zs_zsph))
        for i in range(len(z_up)):
            # sampling points between z_+ and z_-
            z_close = np.linspace(z_low[i], z_up[i], len(z_l))
            # calucalte rightmost integral
            num_integral1[i,:] = scipy.integrate.simps(integrand1, z_close)

        # calculate rightmost integral on denominator using same integrand as before
        # but over the full z_s range 
        integral1 = num_integral1 / scipy.integrate.simps(integrand1, z_s)

        # multiply solution to first integral by weights and integrate over z_ph
        # to find second integral
        integrand2 = weights * integral1
        integral2 = scipy.integrate.simps(integrand2, zs_ph) / scipy.integrate.simps(weights * np.ones(len(zs_ph)), zs_ph)

        # multiply solution to second integral by dndz_l and integrate over z_l
        # to find full numerator and denominator
        integrand3 = dndz_l * integral2
        integral3 = scipy.integrate.simps(integrand3, z_l) / scipy.integrate.simps(dndz_l, z_l)
        
        F = integral3
    
    return F

def get_xi_gg(maxPi, minPi, onehalo=True, year=survey_year):
    '''Get 3D galaxy clustering correlation function from CCL using r vector constructed
    from L.O.S and projected separations'''

    # get spec-z for lenses
    z_l, dndz_l, zleff = zed.get_dndz_spec(gtype='lens', year=year)

    # convert redshifts for lenses and sources to scale factors
    La_arr = 1. / (1. + z_l)
    
    # convert to Mpc/h for definition with Pi
    r_p = setup.arcmin_to_rp(theta_cents, zleff, zed.cosmo_SRD) * (pa.HH0 / 100.)
    
    # define max and min r based on combination of rp and Pi
    rmax = np.sqrt(np.max(r_p)**2 + (maxPi+1)**2) # WE NEED UNITS TO BE CONSISTENT HERE
    rmin = np.sqrt(np.min(r_p)**2 - minPi**2)

    # set some arbitrary 3D separations r to compute xi_gg
    r = np.logspace(np.log10(rmin), np.log10(rmax), interp_len) # Mpc/h

    # wavenumbers and scale factors for all-space power spectrum
    k_arr = np.geomspace(1E-4, 5E2, 3*interp_len)
    
    # must flip array because ccl wants scale factors ascending (z_0 == a_1)
    pk_gg2D = halo.get_Pk2D('gg', k_arr=k_arr, a_arr=np.flip(La_arr), onehalo=onehalo)
    print('Pk_gg calculated')

    # use 2D power spectrum and lens-source 3D separations to get 3D correlation function xi_ls(r,z_l)
    xi_gg = np.zeros([len(r), len(z_l)])
    for zl in range(len(z_l)):
        # r is in Mpc/h so we need to convert back to h
        xi_gg[:,zl] = ccl.correlations.correlation_3d(zed.cosmo_SRD, np.flip(La_arr[zl]), 
                                           r / (pa.HH0 / 100.), pk_gg2D)
    print('xi_gg calculated')
    
    return xi_gg, r_p, r

def get_xi_ls(year=survey_year, onehalo=True):
    '''Interpolate galaxy clustering correlation function over rp and pi to obtain lens-source correlation as a function of these parameters'''

    # get spec-z for sources and lenses
    z_s, dndz_s, zseff = zed.get_dndz_spec(gtype='source', year=year)
    
    # lens dist has been cut and renormalised
    z_l, dndz_l, zleff = zed.get_dndz_spec(gtype='lens', year=year)

    # convert redshifts for lenses and sources to scale factors
    La_arr = 1. / (1. + z_l)
    Sa_arr = 1. / (1. + z_s)

    # comoving radial distance from observer to sources and lenses
    chi_lens = ccl.comoving_radial_distance(zed.cosmo_SRD, La_arr) * (pa.HH0 / 100.)
    chi_source = ccl.comoving_radial_distance(zed.cosmo_SRD, Sa_arr) * (pa.HH0 / 100.)

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
    xi_gg, r_p, r = get_xi_gg(maxPiPos, minPiPos, year=year, onehalo=onehalo)
    
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

    # preallocate list for outputs
    xi_ls = [0] * len(z_l)

    # set first set of indices to be passed to multiprocessing and computed 
    iterables = list(range(len(z_l)))

    # open multiprocess pool in with statement so it closes after it's complete
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
  
    print('xi_ls estimation complete')
    
    com_Pi = [0]*len(z_l)
    z_Pi = [0]*len(z_l)
    # get comoving distance values associated to Pi
    for zl in range(len(z_l)):
        com_Pi[zl] = chi_flipped[zl] + Pi[zl]
        # CCL wants distance in Mpc but we are working in Mpc/h
        z_Pi[zl] = (1./ccl.scale_factor_of_chi(zed.cosmo_SRD, com_Pi[zl] / (pa.HH0_t/100.))) - 1. 

    del com_Pi

    return xi_ls, r_p, Pi, z_Pi, z_l, dndz_l

def get_boosts(year=survey_year, onehalo=True):
    '''Integrate xi_ls over z_spec, z_photo, & z_lens to get boosts as a function of
    rp (or theta since rp is directly defined from theta)'''
    
    start = time.time()
    
    # get weights
    weights = zed.get_weights()
    
    # call to previous function to obtain xi_ls for integral
    xi_ls, r_p, Pi, z_Pi, z_l, dndz_l = get_xi_ls(year=year, onehalo=onehalo)
    
    # set dndz parameters from LSST SRD
    if year == 1:
        z0 = zed.y1_z0_source
        alpha = zed.y1_alp_source
    elif year == 10:
        z0 = zed.y10_z0_source
        alpha = zed.y10_alp_source
    else: 
        print('Not a current survey year')
        exit()
        
    # get source redshift data    
    z_s, dndz_s, zseff = zed.get_dndz_spec('source', year)

    # start by constructing dndz_Pi using SRD parameterisation
    z_Pi_norm = [0] * len(z_l)
    dndz_Pi = [0] * len(z_l)
    dndz_Pi_norm = [0] * len(z_l)
    zph_norm = [0] * len(z_l)

    for zi in range(len(z_l)):
        # un-normalised
        dndz_Pi[zi] = z_Pi[zi]**2 * np.exp(-(z_Pi[zi]/z0) ** alpha)
        
        # dist and redshifts for normalisation
        z_Pi_norm[zi] = np.linspace(np.min(z_s), np.max(z_s)+0.5, len(z_Pi[zi]))
        dndz_Pi_norm[zi] = z_Pi_norm[zi]**2 * np.exp(-(z_Pi_norm[zi]/z0) ** alpha)
    
        dndz_Pi_norm[zi] = dndz_Pi_norm[zi] / scipy.integrate.simps(dndz_Pi_norm[zi], z_Pi_norm[zi]) 
        
        # set full range of photo-z to normalise over
        zph_norm[zi] = np.linspace(np.min(z_s), np.max(z_s)+0.5, len(z_Pi[zi]))

    # next get error dist p_zPi_zph
    p_zPi_zph = [0] * len(z_l)
    p_zPi_zph_norm = [0] * len(z_l)
    
    # first loop over z_Pi values for each lens redshift
    for zl in range(len(z_l)):
        
        # define photo-z shift value for each zl
        chi_plus = ccl.comoving_radial_distance(zed.cosmo_SRD, 
                                            1./(1. + np.max(z_Pi[zl]))) * (pa.HH0/100.) + np.max(Pi[zl])
        a_plus = ccl.scale_factor_of_chi(zed.cosmo_SRD, chi_plus / (pa.HH0/100.))
        zph_plus = (1. / a_plus) - 1.
        # set arbitrary photo-z values in range of z_Pi boosted by 200Mpc in z                                 
        zPi_ph = np.linspace(np.min(z_Pi[zl]), zph_plus, len(z_Pi[zl]))
        
        # variance for each z_Pi
        sig_z = 0.05*(1. + z_Pi[zl])
        sig_z_norm = 0.05*(1. + z_Pi_norm[zl])
        
        p = np.zeros([len(zPi_ph), len(z_Pi[zl])])
        p_norm = np.zeros([len(zPi_ph), len(z_Pi[zl])])
        
        # loop over specific z_Pi values to find error dist for each z_Pi, 
        # also include normalisation over full redshift range
        for zi in range(len(z_Pi[zl])):
            # create 2D array of z_ph probability at each z_Pi
            p[zi,:] =  1. / (np.sqrt(2. * np.pi) * sig_z) * np.exp(-((zPi_ph - z_Pi[zl][zi])**2) / (2. * sig_z**2))
            
            p_norm[zi,:] =  1. / (np.sqrt(2. * np.pi) * sig_z_norm) * np.exp(-((zph_norm[zl] - z_Pi_norm[zl][zi])**2) / (2. * sig_z_norm**2))
            
        # save p(z_Pi,z_ph) at each lens redshift
        p_zPi_zph[zl] = p
        # need to also get error pdf for full zph to normalise over
        p_zPi_zph_norm[zl] = p_norm
        
    
    # define a function to compute the integral over z_s -> relabelled to z_pi
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
            # we normalise over the full spec z range as the slice in z_Pi[zl] only represents 
            # the fact that xi is zero outside this slice
            igl_zPi[ri,:] = scipy.integrate.simps(num_igd_zPi, z_Pi[zl], axis=0) / scipy.integrate.simps(
                denom_igd_zPi, z_Pi_norm[zl])

        # this leaves us with an array that is [len(rp) x len(zph)]
        return igl_zPi
    
    # define function to integrate over z_ph. We do not use z+ and z- as we use the entire soure
    # and lens samples
    global integrate_zph
    def integrate_zph(zl):
        
        # now we want to integrate over zph, so we need array to store integrals for different rp
        igl_zph = np.zeros([len(r_p)])
        
        # arbitrarily define photometric redshifts slightly extended in range from zPi
        chi_plus = ccl.comoving_radial_distance(zed.cosmo_SRD, 
                                            1./(1. + np.max(z_Pi[zl]))) * (pa.HH0/100.) + np.max(Pi[zl])
        a_plus = ccl.scale_factor_of_chi(zed.cosmo_SRD, chi_plus / (pa.HH0/100.))
        zph_plus = (1. / a_plus) - 1. 
        zPi_ph = np.linspace(np.min(z_Pi[zl]), zph_plus, len(z_Pi[zl]))
        
        # call to previous function to get first integral
        igd_zph = weights * integrate_zPi(zl)
        
        # integrate over zph for a certain zl, zph are the columns so specify axis=1, normalise over full photo-z range
        igl_zph = scipy.integrate.simps(igd_zph, zPi_ph, axis=1) / scipy.integrate.simps(weights * np.ones([len(zPi_ph)]), zph_norm[zl])

        # return len(rp) array representing [B(rp) - 1] at zl 
        return igl_zph, zl
    
    # parallelised integration routine
    iterables = list(range(len(z_l)))

    # allocate array to store [B(rp) - 1] for each zl
    igl_zph = np.zeros([len(z_l), len(r_p)])

    # parallelise previous 2 integrals to save run time
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

    # All [B(rp) - 1]_zl have been collected into a [len(z_l) x len(r_p)] array 
    # We must flip along the rows as previous integrals were collected with zl descending, 
    # but we integrate over an ascending dndz_l
    igd_zl = np.flip(igl_zph, axis=0) * dndz_l[:,None]
    
    # compute final integral, which represents B - 1
    igl_zl = scipy.integrate.simps(igd_zl, z_l, axis=0) / scipy.integrate.simps(dndz_l, z_l)
    
    # save data
    np.savez(data_dir+'Boost_data_year%d'%year, B_min1 = igl_zl, B = igl_zl + 1., rp = r_p, theta = theta_cents)

    end = time.time()
    print('Boost estimation complete')
    print('Runtime = %g seconds'%(end-start))
    
    return igl_zl, theta_cents

########### FUNCTIONS BELOW ARE USED FOR THEORETICAL LENSING & IA CALCULATIONS #############

def lens_bias(z_l, year=survey_year):
    
    if year == 1:
        b_z = y1_lens_b / ccl.growth_factor(zed.cosmo_SRD, 1./(1. + z_l)) #normed by ccl
    elif year == 10:
        b_z = y10_lens_b / ccl.growth_factor(zed.cosmo_SRD, 1./(1. + z_l))
    else:
        print('Not a survey year')
        b_z = 0.
    
    return b_z

def get_LSST_Cl(ell, corr, year=survey_year):
    
    lens_z, lens_dndz, zleff = zed.get_dndz_phot(gtype='lens', year=year)
    source_z, source_dndz, zseff = zed.get_dndz_phot(gtype='source', year=year)
    
    a_l = 1. / (1. + lens_z)
    a_s = 1. / (1. + source_z)
      
    # init tracers for lenses and sources
    lensTracer = ccl.NumberCountsTracer(zed.cosmo_SRD, has_rsd=False, dndz=(lens_z, lens_dndz), 
                                    bias=(lens_z, lens_bias(lens_z, year)))
    shearTracer = ccl.WeakLensingTracer(zed.cosmo_SRD, (source_z, source_dndz))
    sourceTracer = ccl.NumberCountsTracer(zed.cosmo_SRD, has_rsd=False, dndz=(source_z, source_dndz),
                                         bias=(source_z, np.ones(len(source_z))))
    
    # defined to be same as in boost function
    k_arr = np.geomspace(1E-4, 5E2, 3*interp_len)
    
    # create 2D power spectrum objects for galaxy-shear correlation
    
    pk_2D = halo.get_Pk2D(corr, k_arr, np.flip(a_l))
    
    # need to make sure pk_gMf is passed in!
    if corr == 'gg':
        aps = ccl.angular_cl(zed.cosmo_SRD, lensTracer, sourceTracer, ell, p_of_k_a=pk_2D)
        tracer1 = lensTracer
        tracer2 = sourceTracer
    elif corr == 'gm':
        aps = ccl.angular_cl(zed.cosmo_SRD, lensTracer, shearTracer, ell, p_of_k_a=pk_2D)
        tracer1 = lensTracer
        tracer2 = shearTracer
    elif corr == 'mm':
        aps = ccl.angular_cl(zed.cosmo_SRD, shearTracer, shearTracer, ell, p_of_k_a=pk_2D)
        tracer1 = shearTracer
        tracer2 = shearTracer
        
    return aps, tracer1, tracer2
    
def get_gammat_noIA(aps, theta, ell):
    '''Calculate gamma_t without IA with theta provided in arcmin'''
    
    gammat = ccl.correlation(zed.cosmo_SRD, ell, aps, theta / 60., type='NG')
    
    return gammat




# --------------------------------------------------------------------------
# This function is a separate get boosts which will test different convolutions
# to find the correct method

def test_boosts(year=survey_year, onehalo=True):
    '''Integrate xi_ls over z_spec, z_photo, & z_lens to get boosts as a function of
    rp (or theta since rp is directly defined from theta)'''
    
    start = time.time()
    
    # get weights
    weights = zed.get_weights()
    
    # call to previous function to obtain xi_ls for integral
    xi_ls, r_p, Pi, z_Pi, z_l, dndz_l = get_xi_ls(year=year, onehalo=onehalo)
    
    # set dndz parameters from LSST SRD
    if year == 1:
        z0 = zed.y1_z0_source
        alpha = zed.y1_alp_source
    elif year == 10:
        z0 = zed.y10_z0_source
        alpha = zed.y10_alp_source
    else: 
        print('Not a current survey year')
        exit()
        
    # get full source redshift data    
    z_s, dndz_s, zseff = zed.get_dndz_spec('source', year)

    # preallocate arrays for integral data
    zpi_full = [0] * len(z_l)
    zph_full = [0] * len(z_l)
    zph_lims = [0] * len(z_l)
    
    # this is dndz_pi * p(zpi,zph) for each zl
    igd_1 = [0] * len(z_l)
    
    # define all these vectors at different lens redshifts
    for zl in range(len(z_l)):
        
        # set full range of spec-z for normalisation of dndz_pi
        zpi_full[zl] = np.linspace(np.min(z_s), np.max(z_s), len(z_Pi[zl]))
        
        # call to get_phot() to get full z_ph vector and dndz_pi * p(zs,zph) 
        zph_full[zl], igd_1[zl], *_ = zed.get_dndz_phot(gtype='source',
                                                          zsmin=np.min(z_Pi[zl]),
                                                          zsmax=np.max(z_Pi[zl]),
                                                          z_vec_len=len(z_Pi[zl]),
                                                          for_Pi=True
                                                         )
        
        # define photo-z shift values for each zl for photo-z integral limits
        chi_plus = ccl.comoving_radial_distance(zed.cosmo_SRD, 
                                            1./(1. + np.max(z_Pi[zl]))) * (pa.HH0/100.) + np.max(Pi[zl])
        a_plus = ccl.scale_factor_of_chi(zed.cosmo_SRD, chi_plus / (pa.HH0/100.))
        zph_plus = (1. / a_plus) - 1.
        
        chi_minus = ccl.comoving_radial_distance(zed.cosmo_SRD, 
                                            1./(1. + np.max(z_Pi[zl]))) * (pa.HH0/100.) - np.max(Pi[zl])
        a_minus = ccl.scale_factor_of_chi(zed.cosmo_SRD, chi_minus / (pa.HH0/100.))
        zph_minus = (1. / a_minus) - 1.
        
        # set arbitrary photo-z values in range of z_Pi boosted by 200Mpc in z
        # these represent the limits of our photo-z integral
        zph_lims[zl] = np.linspace(zph_minus, zph_plus, len(z_Pi[zl]))

    
    # define a function to compute the integral over z_s -> relabelled to z_pi
    def integrate_zPi(zl):
        
        # preallocate arrays to store data
        igl_zPi = np.zeros([len(r_p), len(z_Pi[zl])])
        num_igd = np.zeros([len(z_Pi[zl]), len(z_Pi[zl])])
        
        for ri in range(len(r_p)):
            
            # rows are rp, columns represent different Pi values
            num_igd_zPi = xi_ls[zl][ri,:] * igd_1[zl]
            denom_igd_zPi = igd_1[zl]
            
            # for P(zpi,zph), we want to integrate over the z_Pi values, these are rows so axis=0
            # we normalise over the full spec z range as the slice in z_Pi[zl] only represents 
            # the fact that xi is zero outside this slice
            igl_zPi[ri,:] = scipy.integrate.simps(num_igd_zPi, z_Pi[zl], axis=0) / scipy.integrate.simps(
                denom_igd_zPi, zpi_full[zl])

        # this leaves us with an array that is [len(rp) x len(zph)]
        return igl_zPi
    
    # define function to integrate over z_ph. We do not use z+ and z- as we use the entire soure
    # and lens samples
    global integrate_zph
    def integrate_zph(zl):
        
        # now we want to integrate over zph, so we need array to store integrals for different rp
        igl_zph = np.zeros([len(r_p)])
        
        # call to previous function to get first integral
        igd_zph = weights * integrate_zPi(zl)
        
        # integrate over zph for a certain zl, zph are the columns so specify axis=1, 
        # normalise over lims too (see L2018 paper)
        igl_zph = scipy.integrate.simps(igd_zph, zph_lims[zl], axis=1) / scipy.integrate.simps(weights * np.ones([len(zph_lims[zl])]), zph_lims[zl])

        # return len(rp) array representing [B(rp) - 1] at zl 
        return igl_zph, zl
    
    # parallelised integration routine
    iterables = list(range(len(z_l)))

    # allocate array to store [B(rp) - 1] for each zl
    igl_zph = np.zeros([len(z_l), len(r_p)])

    # parallelise previous 2 integrals to save run time
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

    # All [B(rp) - 1]_zl have been collected into a [len(z_l) x len(r_p)] array 
    # We must flip along the rows as previous integrals were collected with zl descending, 
    # but we integrate over an ascending dndz_l
    igd_zl = np.flip(igl_zph, axis=0) * dndz_l[:,None]
    
    # compute final integral, which represents B - 1
    igl_zl = scipy.integrate.simps(igd_zl, z_l, axis=0) / scipy.integrate.simps(dndz_l, z_l)
    
    # save data
    np.savez(data_dir+'Boost_data_year%d'%year, B_min1 = igl_zl, B = igl_zl + 1., rp = r_p, theta = theta_cents)

    end = time.time()
    print('Boost estimation complete')
    print('Runtime = %g seconds'%(end-start))
    
    return igl_zl, theta_cents
