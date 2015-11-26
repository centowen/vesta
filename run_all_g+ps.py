import numpy as np
import os
import vesta
from units import arcsec

models = ['gaussian', 'gaussian_ps', 'ps', 'disk_ps']
samples = ['k20', 'sbzk', 'ero', 'drg']
subsamples = ['', 'z1']

flux_ext = {}
flux_ps = {}
sigma = {}

flux_ext['gaussian'] = {}
flux_ext['gaussian']['k20']  = {'': 1.14e-3, 'z1': 1.85e-3}
flux_ext['gaussian']['sbzk'] = {'': 2.33e-3, 'z1': 2.34e-3}
flux_ext['gaussian']['ero']  = {'': 1.41e-3, 'z1': 1.49e-3}
flux_ext['gaussian']['drg']  = {'': 2.27e-3, 'z1': 2.47e-3}

flux_ps['gaussian'] = {}
flux_ps['gaussian']['k20']  = {'': 0.00e-3, 'z1': 0.00e-3}
flux_ps['gaussian']['sbzk'] = {'': 0.00e-3, 'z1': 0.00e-3}
flux_ps['gaussian']['ero']  = {'': 0.00e-3, 'z1': 0.00e-3}
flux_ps['gaussian']['drg']  = {'': 0.00e-3, 'z1': 0.00e-3}

sigma['gaussian'] = {}
sigma['gaussian']['k20']  = {'': 0.7/2.35*arcsec, 'z1': 1.0*arcsec/2.35}
sigma['gaussian']['sbzk'] = {'': 0.7/2.35*arcsec, 'z1': 0.7*arcsec/2.35}
sigma['gaussian']['ero']  = {'': 0.6/2.35*arcsec, 'z1': 0.7*arcsec/2.35}
sigma['gaussian']['drg']  = {'': 0.7/2.35*arcsec, 'z1': 0.7*arcsec/2.35}

flux_ext['gaussian_ps'] = {}
flux_ext['gaussian_ps']['k20']  = {'': 0.74e-3, 'z1': 1.84e-3}
flux_ext['gaussian_ps']['sbzk'] = {'': 1.51e-3, 'z1': 1.54e-3}
flux_ext['gaussian_ps']['ero']  = {'': 0.76e-3, 'z1': 0.96e-3}
flux_ext['gaussian_ps']['drg']  = {'': 2.27e-3, 'z1': 2.47e-3}

flux_ps['gaussian_ps'] = {}
flux_ps['gaussian_ps']['k20']  = {'': 0.81e-3, 'z1': 1.00e-3}
flux_ps['gaussian_ps']['sbzk'] = {'': 0.99e-3, 'z1': 1.01e-3}
flux_ps['gaussian_ps']['ero']  = {'': 1.03e-3, 'z1': 1.01e-3}
flux_ps['gaussian_ps']['drg']  = {'': 0.00e-3, 'z1': 0.00e-3}

sigma['gaussian_ps'] = {}
sigma['gaussian_ps']['k20']  = {'': 3.3/2.35*arcsec, 'z1': 3.0*arcsec/2.35}
sigma['gaussian_ps']['sbzk'] = {'': 1.2/2.35*arcsec, 'z1': 1.1*arcsec/2.35}
sigma['gaussian_ps']['ero']  = {'': 2.3/2.35*arcsec, 'z1': 2.4*arcsec/2.35}
sigma['gaussian_ps']['drg']  = {'': 0.7/2.35*arcsec, 'z1': 0.7*arcsec/2.35}


datapath = '/data2/lindroos/ecdfs/aless/stack'
# vis = {}
model='gaussian_ps'

# for sample in samples:
#     for subsample in subsamples:

for sample in ['drg']:
    for subsample in ['z1']:

        vis = os.path.join(datapath, sample+subsample, 'stack.uv.ms')

        parameters, chi2 = \
            vesta.run(vis=vis, model=model,
                    flux_ext=flux_ext[model][sample][subsample], dflux_ext = 0.9e-3,
                    flux_ps=flux_ps[model][sample][subsample], dflux_ps = 0.9e-3,
                    sigma=sigma[model][sample][subsample], dsigma=0.40*arcsec,
                    x=0., y=0., nscan = 20)

        np.save('results/{}_{}_{}_parameters.npy'.format(sample, subsample, model),
                parameters)
        np.save('results/{}_{}_{}_chi2.npy'.format(sample, subsample, model),
                chi2)

        

