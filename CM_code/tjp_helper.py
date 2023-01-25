def generate_config(cosmo, sacc_file, cov_type, Nlen, Nsrc, sig_e, len_bias, IA=None, add_keys=None):
    
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