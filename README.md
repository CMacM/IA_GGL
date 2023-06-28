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

Users should refer to the setup.py files of TJPCov and WeakLensingDeblending to ensure required packages are installed. At the time of writing this readme, the IA halo profile should have been
integrated into the latest release of CCL. However, at the time of writing the code, this was not the case. I am keeping the IA_halo_CCL file for completeness, but it may not be neccessary for
future use of the code in this repository. As this code is not intended for general use, some of the anlysis parameters were hardcoded. They can be modifed in the .py files contained in CM_code. Other
parameters can be directly controlled through function arguements.

### Internal data

Most of the data produced in this analysis can be found in */generated_data* in the form of .npz files. To repeat the analysis that was done in the paper, this data should be all
that is required. Any modifications to boost factors or F will require the associated functions in *spurious_george.py* to be re-run with the desired parameters changed. Modifications to the
galaxy size cut will required additional data which cannot be found in this repo. Please see below.

### External data

To produce the LSST WeakLensingDeblending catalogues used to account for the galaxy size cut, a OneDegSq.fits file will be required.
As far as I am aware, this is only available on NERSC. Please contact me directly if this is needed and I can provide you with this file. Alternatively, 
I can provide the LSST cataloguese themselves upon request.


