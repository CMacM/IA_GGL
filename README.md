This repo contains the code used to produce the results of Section 4 in https://arxiv.org/abs/2306.11428

**Disclaimer**: This code is for a highly specfiic scientific analysis and is not intended for wide-scale use. 
I have tried to clearly detail its setup here, but in places it is still messy and may be convoluted to use, as it was not
intended to be run many times by different users.

### Code Contents

The code makes use of five external repositories: TJPcov (*/tjpcov*), c-d-leonard/IA_GGL (*/DL_basis_code*), LSSTDESC/Requirements (*/WLD_size_cutting/Requirements*), 
LSSTDESC/WeakLensingDeblending (*/WLD_size_cutting/WeakLensingDeblending*), and IA_halo_CCL (*/IA_halo_CCL*). CM_code contains code written or adapted from other sources for the specific
requirements of this analysis.

In CM_code, there are various .py files which contain functions to perform specific parts of the analysis. */lsst_coZmology.py* contains the fiducial cosmologies
and functions to create redshift distributions and perform various redshift-related actions. *halo_model.py* contains classes for the source and lens halo occupation
distributions, as well as functions related to setting up the halo model and obtaining power spectra from it. *spurious_george.py* contains functions to carry out
top level calculations, such as for the boost factors and F. *tjp_helper.py* contains functions which make use of TJPcov to generate covariance matrices specific
to this analysis.

Jupyter notebooks contain analysis and testing done using the fundamental code outlined above. Those labelled analysis have been used to generate the results presented in the
paper, whereas those labelled testing were used along the way to help build and validate functions found in CM_code, as well as some older analysis which was altered for the
paper, but is being kept in case it is useful in future.

### Running the code (June 2023, subject to change)

Because of the various code sources used, running the code can be complex. It is advisable that two environments are created, one which contains the latest release version of CCL alongside
TJPCov and and all its dependencies, as well as any other packages required by CM_code or the analysis notebooks.

**For anything involving the IA halo model** a second environment will be needed. This should contain a developer install of CCL specific to the IA_halo_model branch. Additional packages in
this environment should be kept to the minimum neccessary.

Any user will need to determine how to use the functions in CM_code for a modified analysis from the source code itself, or contact me. However, to reproduce the
original analysis, the notebooks will be able to run using the data stored in */generated_data* given the correct packages are installed.

### Internal data

Most of the data produced in this analysis can be found in */generated_data* in the form of .npz files. To repeat the analysis as done in the paper, this should be all
that is required.

### External data

If attempting to reproduce this analysis from scratch, additional external data will be neccessary. To produce LSST WeakLensingDeblending catalogues, the OneDegSq.fits file will be required.
As far as I am aware, this is only available on NERSC. Please contact me directly if this is needed and I can provide you with this file. Additionally, the resulting LSST catalogues are not
contained in this repo due to their size. I can also provide these upon request.


