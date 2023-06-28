import numpy as np
import sacc
import matplotlib.pyplot as plt
import pyccl as ccl
import scipy
import importlib as imp
import astropy.io.fits as fits
import time

import CM_code.spurious_george as sp
imp.reload(sp)

# import main.py from TJPCov folder
import tjpcov as tjp
from tjpcov.covariance_calculator import CovarianceCalculator

datdir = '/home/b7009348/WGL_project/LSST_forecast_code/generated_data/'

# set up default parameters
survey_year = sp.survey_year

def generate_config(cosmo, sacc_file, cov_type, Nlen, Nsrc, sig_e, len_bias, IA=None, add_keys=None):
    """This function is used to generate configuration dictionaries for TJPCov.
    Required arguements are neccerssary for TJP to run. Additonal parameters can be
    passed as a dictionary to the add_keys arguement"""
    
    config = {}
    config['tjpcov'] = {}
    config['tjpcov']['cosmo'] = cosmo
    config['tjpcov']['sacc_file'] = sacc_file
    config['tjpcov']['cov_type'] = cov_type
    config['tjpcov']['Ngal_lens'] = Nlen
    config['tjpcov']['Ngal_source'] = Nsrc
    config['tjpcov']['sigma_e_source'] = sig_e
    config['tjpcov']['bias_lens'] = len_bias
    config['tjpcov']['IA'] = IA
    
    if add_keys is not None:
        
        for i in range(len(add_keys)):
            
            if add_keys[i][0] not in config:
                
                config[add_keys[i][0]] = {}
                
            for j in range(len(add_keys[i])-1):
                
                config[add_keys[i][0]][add_keys[i][j+1][0]] = add_keys[i][j+1][1]
    
    return config


def create_xi_t_sacc(zlmin, zlmax, zsmin, zsmax, year=survey_year):
    # create LSST sacc file
    s = sacc.Sacc()

    # start by adding redshift data (this will be used internally by TJP
    # to create WL and NC tracers
    _, _, zleff = sp.zed.get_dndz_spec(gtype='lens', zlmin=zlmin, zlmax=zlmax, year=year)
    
    z_l, dndz_l, _ = sp.zed.get_dndz_phot(gtype='lens', zlmin=zlmin, zlmax=zlmax, year=year)
    s.add_tracer('NZ','lens', z=z_l, nz=dndz_l)

    z_s, dndz_s, _ = sp.zed.get_dndz_phot(gtype='source', zsmin=zsmin, zsmax=zsmax, year=year)
    s.add_tracer('NZ','source', z=z_s, nz=dndz_s)
    
    r_p = np.logspace(np.log10(sp.rpmin),np.log10(sp.rpmax),sp.N_bins)
    
    th_lsst = sp.rp_to_arcmin(rp=r_p,zeff=zleff)
    
    print('rp:', r_p)
    print('th:', th_lsst)

    lensTracer = ccl.NumberCountsTracer(sp.zed.cosmo_SRD, has_rsd=False, dndz=(z_l, dndz_l), 
                                        bias=(z_l, sp.lens_bias(z_l,year=year))
                                       )
    shearTracer = ccl.WeakLensingTracer(sp.zed.cosmo_SRD, (z_s, dndz_s), has_shear=True,
                                        ia_bias=None
                                       )

    ell = np.unique(np.geomspace(180/(th_lsst[-1]/60.) - 10, 180/(th_lsst[0]/60.) + 10, 1024).astype(int))

    Cl = ccl.angular_cl(sp.zed.cosmo_SRD, lensTracer, shearTracer, ell)

    LSST_xi = ccl.correlation(sp.zed.cosmo_SRD, ell, Cl, th_lsst/60., type='NG') # take theta in degrees

    s.add_theta_xi('galaxy_shearDensity_xi_t', 'lens', 'source', th_lsst, LSST_xi) # save in arcmins

    s.save_fits(datdir+'forecast_IA_data_lsst-y%d.fits'%year, overwrite=True)
    
    return

def get_lsst_covariance(zlmin, zlmax, zsmin, zsmax, lmax, rho, year=survey_year, out_file=None):
    
    start = time.time()
    
    # find fraction of lens sample in chosen bin 
    zl_full, dndzl_full, *_ = sp.zed.get_dndz_phot(gtype='lens', zlmin=0.05, zlmax=3.5, year=year, normalise=False)
    
    zl, dndzl, *_ = sp.zed.get_dndz_phot(gtype='lens', zlmin=zlmin, zlmax=zlmax, year=year, normalise=False)
    
    lens_tot = scipy.integrate.simps(dndzl_full, zl_full)
    lens_bin = scipy.integrate.simps(dndzl, zl)
    
    lens_frac = lens_bin/lens_tot
    print('Fraction of lenses in bin: %g'%lens_frac)
    
    # find fraction of source sample in chosen bin 
    zs_full, dndzs_full, *_ = sp.zed.get_dndz_phot(gtype='source', zsmin=0.05, zsmax=3.5, year=year, normalise=False)
    
    zs, dndzs, *_ = sp.zed.get_dndz_phot(gtype='source', zsmin=zsmin, zsmax=zsmax, year=year, normalise=False)
    
    source_tot = scipy.integrate.simps(dndzs_full, zs_full)
    source_bin = scipy.integrate.simps(dndzs, zs)
    
    source_frac = source_bin/source_tot
    print('Fraction of source in bin: %g'%source_frac)
    
    # set configuration parameters
    if year == 1:
        N_len = 18.
        N_src = 6.39
    elif year == 10:
        N_len = 48.
        N_src = 13.61
            
    # create configs
    
    b_lens = np.mean(sp.lens_bias(z_l=zl, year=year))
    
    y1y1_config = generate_config(cosmo=sp.zed.cosmo_SRD, 
                                  sacc_file=datdir+'forecast_IA_data_lsst-y%d.fits'%year,
                                  cov_type='RealGaussianFsky', 
                                  Nlen=lens_frac*N_len, 
                                  Nsrc=source_frac*N_src, 
                                  sig_e=0.26, 
                                  len_bias=b_lens, 
                                  IA=0.8,
                                  add_keys=[['ProjectedReal',['lmax', lmax]],
                                            ['GaussianFsky',['fsky', 18000/41253]]
                                           ]
                                 )
    
    # get covariance for same estimator y1y1
    print('Getting y1,y1 covariance...')
    print('')
    
    cov_y1y1 = CovarianceCalculator(y1y1_config).get_covariance()
    
    print('')
    print('Found covariance matrix for y1y1, getting y1,y2 for different rho...')
    print('')
    
    # loop over different rho values
    full_cov = []
    
    for value in rho:
    
        y1y2_config = generate_config(cosmo=sp.zed.cosmo_SRD, 
                                      sacc_file=datdir+'forecast_IA_data_lsst-y%d.fits'%year,
                                      cov_type='RealGaussianFsky', 
                                      Nlen=lens_frac*N_len, 
                                      Nsrc=source_frac*N_src, 
                                      sig_e=np.sqrt(value*0.26*0.26), 
                                      len_bias=b_lens, 
                                      IA=0.8,
                                      add_keys=[['ProjectedReal',['lmax', lmax]],
                                                ['GaussianFsky',['fsky', 18000/41253]]
                                               ]
                                     )
        
        cov_y1y2 = CovarianceCalculator(y1y2_config).get_covariance()
        
        full_cov.append(2*cov_y1y1 - 2*cov_y1y2)
        
        print('')
        print('Getting cov[y1,y2] for rho=%g'%value)
        
    if out_file is None:
        out_file = 'lsst-y%d-covmats_rho=%g-%g'%(year,rho[0],rho[-1])
        
    np.savez(file=datdir+out_file, covs=full_cov, rho=rho)
    
    end = time.time()
    print('Runtime: %g mins' %((end-start)/60))
    
    return full_cov
        