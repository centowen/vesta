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


# Based on scitools meshgrid, copied from numpy 1.8
def meshgrid(*xi, **kwargs):
    """
    Return coordinate matrices from two or more coordinate vectors.
    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.
    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        See Notes for more details.
    sparse : bool, optional
         If True a sparse grid is returned in order to conserve memory.
         Default is False.
    copy : bool, optional
        If False, a view into the original arrays are returned in
        order to conserve memory.  Default is True.  Please note that
        ``sparse=False, copy=False`` will likely return non-contiguous arrays.
        Furthermore, more than one element of a broadcast array may refer to
        a single memory location.  If you need to write to the arrays, make
        copies first.
    Returns
    -------
    X1, X2,..., XN : ndarray
        For vectors `x1`, `x2`,..., 'xn' with lengths ``Ni=len(xi)`` ,
        return ``(N1, N2, N3,...Nn)`` shaped arrays if indexing='ij'
        or ``(N2, N1, N3,...Nn)`` shaped arrays if indexing='xy'
        with the elements of `xi` repeated to fill the matrix along
        the first dimension for `x1`, the second for `x2` and so on.
    Notes
    -----
    This function supports both indexing conventions through the indexing keyword
    argument.  Giving the string 'ij' returns a meshgrid with matrix indexing,
    while 'xy' returns a meshgrid with Cartesian indexing.  In the 2-D case
    with inputs of length M and N, the outputs are of shape (N, M) for 'xy'
    indexing and (M, N) for 'ij' indexing.  In the 3-D case with inputs of
    length M, N and P, outputs are of shape (N, M, P) for 'xy' indexing and (M,
    N, P) for 'ij' indexing.  The difference is illustrated by the following
    code snippet::
        xv, yv = meshgrid(x, y, sparse=False, indexing='ij')
        for i in range(nx):
            for j in range(ny):
                # treat xv[i,j], yv[i,j]
        xv, yv = meshgrid(x, y, sparse=False, indexing='xy')
        for i in range(nx):
            for j in range(ny):
                # treat xv[j,i], yv[j,i]
    See Also
    --------
    index_tricks.mgrid : Construct a multi-dimensional "meshgrid"
                     using indexing notation.
    index_tricks.ogrid : Construct an open multi-dimensional "meshgrid"
                     using indexing notation.
    Examples
    --------
    >>> nx, ny = (3, 2)
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> xv, yv = meshgrid(x, y)
    >>> xv
    array([[ 0. ,  0.5,  1. ],
           [ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]])
    >>> xv, yv = meshgrid(x, y, sparse=True)  # make sparse output arrays
    >>> xv
    array([[ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.],
           [ 1.]])
    `meshgrid` is very useful to evaluate functions on a grid.
    >>> x = np.arange(-5, 5, 0.1)
    >>> y = np.arange(-5, 5, 0.1)
    >>> xx, yy = meshgrid(x, y, sparse=True)
    >>> z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    >>> h = plt.contourf(x,y,z)
    """
    if len(xi) < 2:
        msg = 'meshgrid() takes 2 or more arguments (%d given)' % int(len(xi) > 0)
        raise ValueError(msg)

    args = np.atleast_1d(*xi)
    ndim = len(args)

    copy_ = kwargs.get('copy', True)
    sparse = kwargs.get('sparse', False)
    indexing = kwargs.get('indexing', 'xy')
    if not indexing in ['xy', 'ij']:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [x.reshape(s0[:i] + (-1,) + s0[i + 1::]) for i, x in enumerate(args)]

    shape = [x.size for x in output]

    if indexing == 'xy':
        # switch first and second axis
        output[0].shape = (1, -1) + (1,)*(ndim - 2)
        output[1].shape = (-1, 1) + (1,)*(ndim - 2)
        shape[0], shape[1] = shape[1], shape[0]

    if sparse:
        if copy_:
            return [x.copy() for x in output]
        else:
            return output
    else:
        # Return the full N-D matrix (not only the 1-D vector)
        if copy_:
            mult_fact = np.ones(shape, dtype=int)
            return [x * mult_fact for x in output]
        else:
            return np.broadcast_arrays(*output)


def run(vis, model = 'gaussian', flux_ext=1e-3, dflux_ext=1e-4,
        flux_ps=0e-3, dflux_ps=1e-4,
        sigma=1*arcsec, dsigma=1*arcsec, x=0., y=0., nscan=10):

    libpath = os.path.join(os.path.abspath(__path__[0]), 'build', 'libchi2_scan.so')
    libchi2 = cdll.LoadLibrary(libpath)

    if model =='gaussian':
        shape = [nscan, nscan]
    else:
        shape = [nscan]*3
    chi2 = np.zeros(shape)

    xs = np.zeros(shape)
    ys = np.zeros(shape)

    if model == 'gaussian':
        fluxes_ext = np.linspace(flux_ext-dflux_ext, flux_ext+dflux_ext, nscan)
        sigmas = np.linspace(sigma-dsigma, sigma+dsigma, nscan)
        fluxes_ext,sigmas = np.meshgrid(fluxes_ext, sigmas)

        parameters = np.zeros(shape+[4])
        parameters[:,:,0] = fluxes_ext
        parameters[:,:,1] = xs
        parameters[:,:,2] = ys
        parameters[:,:,3] = sigmas
    elif model == 'gaussian_ps':
        fluxes_ext = np.linspace(flux_ext-dflux_ext, flux_ext+dflux_ext, nscan)
        fluxes_ps = np.linspace(flux_ps-dflux_ps, flux_ps+dflux_ps, nscan)
        sigmas = np.linspace(sigma-dsigma, sigma+dsigma, nscan)
        fluxes_ext, fluxes_ps, sigmas = meshgrid(fluxes_ext, fluxes_ps, sigmas, indexing='xy')

        parameters = np.zeros(shape+[5])
        parameters[:,:,:,0] = fluxes_ext
        parameters[:,:,:,1] = xs
        parameters[:,:,:,2] = ys
        parameters[:,:,:,3] = sigmas
        parameters[:,:,:,4] = fluxes_ps
#         fluxes,sigmas = np.meshgrid(fluxes, sigmas)
    

    chi2_scan = libchi2.c_chi2_scan
    c_chi2 = c_ndarray(chi2, dtype=chi2.dtype, ndim=chi2.ndim)
    c_parameters = c_ndarray(parameters, dtype=parameters.dtype, ndim=parameters.ndim)
    chi2_scan(c_chi2, c_int(chi2.ndim), c_char_p(vis), c_model[model], c_parameters)
    return parameters, chi2


if __name__ == '__main__':
    import vesta
    parameters, chi2 = \
        vesta.run(vis = '/data2/lindroos/ecdfs/aless/stack/sbzk/stack.uv.ms',
                  flux = 2.3349e-3, dflux = 0.6e-3,
                  sigma=0.30536*arcsec, dsigma=0.16*arcsec,
                  x=1.489274040147295e-07, y=9.887614358626681e-09,
                  nscan = 20)

    np.save('results/sbzk_parameters.npy', parameters)
    np.save('results/sbzk_chi2.npy', chi2)
