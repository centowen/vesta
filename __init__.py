import numpy as np
from ctypes import cdll, c_int, c_char_p
from numpyctypes import c_ndarray
import os
from units import arcsec

c_model = {'gaussian': 0,
           'gaussian_ps': 1,
           'ps': 2,
           'disk': 3,
           'disk_ps': 4}



def run(vis = '/data2/lindroos/ecdfs/aless/stack/sbzk/stack.uv.ms',
        model = 'gaussian'):

    libpath = os.path.join(os.path.abspath(__path__[0]), 'build', 'libchi2_scan.so')
    libchi2 = cdll.LoadLibrary(libpath)

    shape = [10, 10]
    chi2 = np.zeros(shape)

    x = np.zeros(shape)
    y = np.zeros(shape)
    PA = np.zeros(shape)

    flux = np.linspace(1.5e-3, 2.5e-3, 10)
    sigma = np.linspace(0, 1.*arcsec, 10)
    flux,sigma = np.meshgrid(flux, sigma)
    
    parameters = np.zeros(shape+[6])
    parameters[:,:,0] = flux
    parameters[:,:,1] = x
    parameters[:,:,2] = y
    parameters[:,:,3] = sigma
#     parameters[
#     parameters = np.array([flux, x, y, sigma_x, sigma_y, PA])
#     print(shape+[6])
#     parameters = parameters.swapaxes(0, 1)

    chi2_scan = libchi2.c_chi2_scan
    c_chi2 = c_ndarray(chi2, dtype=chi2.dtype, ndim=chi2.ndim)
    c_parameters = c_ndarray(parameters, dtype=parameters.dtype, ndim=parameters.ndim)
    chi2_scan(c_chi2, c_int(chi2.ndim), c_char_p(vis), c_model[model], c_parameters)
    print(chi2)

if __name__ == '__main__':
    import vesta
    vesta.run()
