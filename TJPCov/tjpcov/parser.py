#Original copied from firecrown.
import importlib
import yaml
import jinja2


def parse(filename):
    """Parse a configuration file.

    Parameters
    ----------
    filename : str
        The config file to parse. Should be YAML formatted.

    Returns
    -------
    config: dict
        The raw config file as a dictionary.
    data : dict
        A dictionary containg each analyses key replaced with its
        corresponding data and function to compute the log-likelihood.
    """

    with open(filename, 'r') as fp:
        config_str = jinja2.Template(fp.read()).render()
    config = yaml.load(config_str, Loader=yaml.Loader)
    data = yaml.load(config_str, Loader=yaml.Loader)

    if 'parameters' in data:
        params = {}
        for p, val in data['parameters'].items():
            if isinstance(val, list) and not isinstance(val, str):
                params[p] = val[1]
            else:
                params[p] = val
        data['parameters'] = params

    analyses = list(
        set(list(data.keys())) -
        set(['parameters', 'cosmosis', 'emcee', 'tjpcov', 'firecrown',
             'two_point', 'priors', 'cache']))

    for analysis in analyses:
        new_keys = {}

        try:
            mod = importlib.import_module(data[analysis]['module'])
        except Exception:
            print("Module '%s' for analysis '%s' cannot be imported!" % (
                data[analysis]['module'], analysis))
            raise

    return config, data