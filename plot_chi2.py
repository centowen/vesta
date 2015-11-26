import numpy as np
import os
import vesta
from units import arcsec
import pylab as pl
# from scipy.interpolate import SmoothBivariateSpline
from scipy.interpolate import RectBivariateSpline


flux = {}
flux['gaussian'] = {}
flux['gaussian']['k20']  = {'': 1.14e-3, 'z1': 1.85e-3}
flux['gaussian']['sbzk'] = {'': 2.33e-3, 'z1': 2.34e-3}
flux['gaussian']['ero']  = {'': 1.41e-3, 'z1': 1.49e-3}
flux['gaussian']['drg']  = {'': 2.27e-3, 'z1': 2.47e-3}
sigma = {}
sigma['gaussian'] = {}
sigma['gaussian']['k20']  = {'': 0.7/2.35*arcsec, 'z1': 1.0*arcsec}
sigma['gaussian']['sbzk'] = {'': 0.7/2.35*arcsec, 'z1': 0.7*arcsec}
sigma['gaussian']['ero']  = {'': 0.6/2.35*arcsec, 'z1': 0.7*arcsec}
sigma['gaussian']['drg']  = {'': 0.7/2.35*arcsec, 'z1': 0.7*arcsec}


models = ['gaussian', 'gaussian_ps', 'ps', 'disk_ps']
samples = ['k20', 'sbzk', 'ero', 'drg']
subsamples = ['', 'z1']


def load_data(sample='sbzk', subsample='', model='gaussian'):
    parameters = np.load('results/{}_{}_{}_parameters.npy'.format(sample, subsample, model))
    chi2       = np.load('results/{}_{}_{}_chi2.npy'.format(sample, subsample, model))

    return parameters, chi2


def comp_fits(sample='sbzk', subsample='', model='gaussian'):
    parameters, chi2 = load_data(sample, subsample, model)

    fluxes = parameters[:,:,0]
    sigmas = parameters[:,:,3]
    fluxes_highres = np.linspace(np.min(fluxes), np.max(fluxes), fluxes.shape[0]*10)
    sigmas_highres = np.linspace(np.min(sigmas), np.max(sigmas), sigmas.shape[1]*10)
    fluxes_highres, sigmas_highres = np.meshgrid(fluxes_highres, sigmas_highres)
    chi2_int = RectBivariateSpline(fluxes[0,:], sigmas[:,0], chi2)
    chi2_highres = chi2_int(fluxes_highres[0,:], sigmas_highres[:,0])


#     new_fit_flux = fluxes[np.unravel_index(np.argmin(chi2), dims=chi2.shape)]
#     new_fit_sigma = sigmas[np.unravel_index(np.argmin(chi2), dims=chi2.shape)]

    new_fit_flux = fluxes_highres[np.unravel_index(np.argmin(chi2_highres), dims=chi2_highres.shape)]
    new_fit_sigma = sigmas_highres[np.unravel_index(np.argmin(chi2_highres), dims=chi2_highres.shape)]
    upper_sigma = np.max(sigmas_highres[chi2_highres < 2.31+np.min(chi2_highres)])
    lower_sigma = np.min(sigmas_highres[chi2_highres < 2.31+np.min(chi2_highres)])
    upper_flux = np.max(fluxes_highres[chi2_highres < 2.31+np.min(chi2_highres)])
    lower_flux = np.min(fluxes_highres[chi2_highres < 2.31+np.min(chi2_highres)])

    print('flux: {:.2f}+{:.2f}-{:.2f}'.format(
                                    new_fit_flux*1e3,
                                    (upper_flux-new_fit_flux)*1e3,
                                    (new_fit_flux-lower_flux)*1e3))
    print('sigma: {:.2f}+{:.2f}-{:.2f}'.format(
                                    2.35*new_fit_sigma/arcsec,
                                    2.35*(upper_sigma-new_fit_sigma)/arcsec,
                                    2.35*(new_fit_sigma-lower_sigma)/arcsec))


def plot_chi2(sample='sbzk', subsample='', model='gaussian'):
    parameters, chi2 = load_data(sample, subsample, model)

    fluxes = parameters[:,:,0]
    sigmas = parameters[:,:,3]

    fluxes_highres = np.linspace(np.min(fluxes), np.max(fluxes), fluxes.shape[0]*10)
    sigmas_highres = np.linspace(np.min(sigmas), np.max(sigmas), sigmas.shape[1]*10)
    fluxes_highres, sigmas_highres = np.meshgrid(fluxes_highres, sigmas_highres)
    chi2_int = RectBivariateSpline(fluxes[0,:], sigmas[:,0], chi2)

    chi2_highres = chi2_int(fluxes_highres[0,:], sigmas_highres[:,0])
    pl.contour(fluxes_highres, sigmas_highres/arcsec,
               chi2_int(fluxes_highres[0,:], sigmas_highres[:,0]).reshape(fluxes_highres.shape),
               np.min(chi2_int(fluxes_highres[0,:], sigmas_highres[:,0]))+np.array([2.3, 4.61, 9.21, 13.82]))


def main():
    for sample in samples:
        for subsample in subsamples:
            print(sample+subsample+':')
            pl.figure()
            pl.title(sample+subsample)
            plot_chi2(sample, subsample)
            comp_fits(sample, subsample)
    pl.show()


if __name__ == '__main__':
    main()
