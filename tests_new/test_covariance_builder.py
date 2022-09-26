import healpy as hp
import os
import pytest
import numpy as np
import sacc
import pickle
import pyccl as ccl
import pymaster as nmt
from tjpcov_new.covariance_builder import CovarianceBuilder


root = "./tests/benchmarks/32_DES_tjpcov_bm/"
outdir = root + 'tjpcov_tmp/'
input_yml = os.path.join(root, "tjpcov_conf_minimal.yaml")
input_sacc = sacc.Sacc.load_fits(root + 'cls_cov.fits')

# Create temporal folder
os.makedirs('tests/tmp/', exist_ok=True)


class CovarianceBuilderTester(CovarianceBuilder):
    # Based on https://stackoverflow.com/a/28299369
    def _build_matrix_from_blocks(self, blocks, tracers_cov):
        super()._build_matrix_from_blocks(blocks, tracers_cov)

    def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
        super().get_covariance_block(tracer_comb1, tracer_comb2, **kwargs)


def get_nmt_bin(lmax=95):
    bpw_edges = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72,
                           78, 84, 90, 96])
    if lmax != 95:
        # lmax + 1 because the upper edge is not included
        bpw_edges = bpw_edges[bpw_edges < lmax+1]
        bpw_edges[-1] = lmax+1

    return  nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])


def test_smoke():
    CovarianceBuilderTester(input_yml)


def test_nuisance_config():
    cb = CovarianceBuilderTester(input_yml)
    assert cb.bias_lens == {'DESgc__0': 1.48}
    assert cb.IA is None
    Ngal = 26 * 3600 / (np.pi / 180)**2
    assert cb.Ngal == {'DESgc__0': Ngal, 'DESwl__0': Ngal, 'DESwl__1': Ngal}
    assert cb.sigma_e == {'DESwl__0': 0.26, 'DESwl__1': 0.26}

# TODO: Add tests for _split_tasks_by_rank, _compute_all_blocks and
# _build_matrix_from_blocks, get_covariance. They are tested through the
# NaMaster and the SSC run but it would be better to have dedicated tests.

def test_split_tasks_by_rank():
    pass


def test_build_matrix_from_blocks_not_implemented():
    with pytest.raises(NotImplementedError):
        cb = CovarianceBuilderTester(input_yml)
        cb._build_matrix_from_blocks([], [])


def test_compute_all_blocks():
    def get_covariance_block(tracer_comb1, tracer_comb2, **kwargs):
        f1 = int(tracer_comb1[0].split('__')[1]) + 1
        f2 = int(tracer_comb1[1].split('__')[1]) + 1
        f3 = int(tracer_comb2[0].split('__')[1]) + 1
        f4 = int(tracer_comb2[1].split('__')[1]) + 1

        block = f1 * f2 * f3 * f4 * np.ones((10, 10))
        return block

    class CovarianceBuilderTester(CovarianceBuilder):
        # Based on https://stackoverflow.com/a/28299369
        def _build_matrix_from_blocks(self, blocks, tracers_cov):
            super()._build_matrix_from_blocks(blocks, tracers_cov)

        def get_covariance_block(self, tracer_comb1, tracer_comb2, **kwargs):
            return get_covariance_block(tracer_comb1, tracer_comb2, **kwargs)


    cb = CovarianceBuilderTester(input_yml)
    blocks, tracers_blocks = cb._compute_all_blocks()

    for bi, trs in zip(blocks, tracers_blocks):
        assert np.all(bi == get_covariance_block(trs[0], trs[1]))


def test_get_cosmology():
    # Check that it reads the parameters from the yaml file
    cb = CovarianceBuilderTester(input_yml)
    config = cb.config.copy()
    assert isinstance(cb.get_cosmology(), ccl.Cosmology)

    # Check that it uses the cosmology if given
    cosmo = ccl.CosmologyVanillaLCDM()
    config['tjpcov']['cosmo'] = cosmo
    cb = CovarianceBuilderTester(config)
    assert cb.get_cosmology() is cosmo

    # Check that it reads a cosmology from a yml file
    config['tjpcov']['cosmo'] = './tests/data/cosmo_desy1.yaml'
    cb = CovarianceBuilderTester(config)
    assert isinstance(cb.get_cosmology(), ccl.Cosmology)

    # Check it reads pickles too
    fname = "tests/tmp/cosmos_desy1.pkl"
    with open(fname, 'wb') as ff:
        pickle.dump(cosmo, ff)

    config['tjpcov']['cosmo'] = fname
    cb = CovarianceBuilderTester(config)
    assert isinstance(cb.get_cosmology(), ccl.Cosmology)

    # Check that any other thing rises an error

    with pytest.raises(ValueError):
        config['tjpcov']['cosmo'] = ['hello']
        cb = CovarianceBuilderTester(config)
        cb.get_cosmology()


def test_get_covariance_block_not_implemented():
    with pytest.raises(NotImplementedError):
        cb = CovarianceBuilderTester(input_yml)
        cb.get_covariance_block([], [])


def test_get_covariance():
    pass


def test_get_list_of_tracers_for_cov():
    cb = CovarianceBuilderTester(input_yml)
    trs_cov = cb.get_list_of_tracers_for_cov()

    # Test all tracers
    trs_cov2 = []
    tracers = cb.io.get_sacc_file().get_tracer_combinations()
    for i, trs1 in enumerate(tracers):
        for trs2 in tracers[i:]:
            trs_cov2.append((trs1, trs2))

    assert trs_cov == trs_cov2


def test_get_mask_names_dict():
    tracer_names = {1: 'DESwl__0', 2: 'DESgc__0', 3: 'DESwl__1', 4: 'DESwl__1'}
    cb = CovarianceBuilderTester(input_yml)
    mn = cb.get_mask_names_dict(tracer_names)

    assert isinstance(mn, dict)
    for i, mni in mn.items():
        tni = tracer_names[i]
        assert mni == cb.config['tjpcov']['mask_names'][tni]


def test_get_masks_dict():
    tracer_names = {1: 'DESwl__0', 2: 'DESgc__0', 3: 'DESwl__1', 4: 'DESwl__1'}
    cb = CovarianceBuilderTester(input_yml)
    m = cb.get_masks_dict(tracer_names)

    assert isinstance(m, dict)
    for i, mni in m.items():
        tni = tracer_names[i]
        assert np.all(mni ==
                      hp.read_map(cb.config['tjpcov']['mask_file'][tni]))

    mi = np.arange(100)
    cache = {f'm{i+1}': mi+i for i in range(4)}
    m = cb.get_masks_dict(tracer_names, cache)
    for i in range(4):
        assert np.all(m[i+1] == mi + i)


def test_get_nbpw():
    cb = CovarianceBuilderTester(input_yml)
    assert cb.get_nbpw() == 16


def test_get_tracers_spin_dict():
    tracer_names = {1: 'DESwl__0', 2: 'DESgc__0', 3: 'DESwl__1', 4: 'DESwl__1'}
    cb = CovarianceBuilderTester(input_yml)
    s = cb.get_tracers_spin_dict(tracer_names)

    assert s == {1: 2, 2: 0, 3: 2, 4: 2}


def test_get_tracer_comb_spin():
    tracer_comb = ['DESwl__0', 'DESgc__0']
    cb = CovarianceBuilderTester(input_yml)
    assert cb.get_tracer_comb_spin(tracer_comb) == (2, 0)


@pytest.mark.parametrize("tr", ['DESwl__0', 'DESgc__0'])
def test_get_tracer_nmaps(tr):
    cb = CovarianceBuilderTester(input_yml)
    nmap = 2 if tr == 'DESwl__0' else 1
    assert cb.get_tracer_nmaps(tr) == nmap


