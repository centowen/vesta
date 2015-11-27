// vesta, use Ceres to douv-model fitting for ms-data.
// Copyright (C) 2015  Lukas Lindroos
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA. 
//
#include "DiskAndDeltaCost.h"

DiskAndDeltaCost::DiskAndDeltaCost(float* u, float* v,
                                   float* V_real, 
                                   float* V_imag,
                                   float* weights, 
                                   int* flags,
                                   const int nchan, 
                                   const int nstokes,
								   const int nrow)
                                  : _nchan(nchan), _nstokes(nstokes), _nrow(nrow)
{
	for(int i = 0; i < 5; i++) {
		mutable_parameter_block_sizes()->push_back(1);
	}

	set_num_residuals(2*_nchan*_nstokes*_nrow);

	_u = new double[_nchan*_nstokes*_nrow];
	_v = new double[_nchan*_nstokes*_nrow];
	_V_real  = new double[_nchan*_nstokes*_nrow];
	_V_imag  = new double[_nchan*_nstokes*_nrow];
	_flags   = new   bool[_nchan*_nstokes*_nrow];
	sqrt_weights = new double[_nstokes*_nrow];

	size_t index;
	for(int uvrow = 0; uvrow < _nrow; uvrow++)
	{
		for(int chan = 0; chan < _nchan; chan++)
		{
			for(int pol = 0; pol < _nstokes; pol++)
			{
				index = _nstokes*_nchan*uvrow+chan+pol*_nchan;
				_u[index]      = (double)u     [index];
				_v[index]      = (double)v     [index];
				_V_real[index] = (double)V_real[index];
				_V_imag[index] = (double)V_imag[index];
				if(flags != NULL)
					_flags[index]  = flags [index];

				if(chan == 0) 
				{
					sqrt_weights[_nstokes*uvrow+pol] =
						sqrt((double)weights[_nstokes*_nchan*uvrow+pol]);
				}
			}
		}
	}
}
DiskAndDeltaCost::~DiskAndDeltaCost()
{
	delete[] _u;
	delete[] _v;
	delete[] _V_real;
	delete[] _V_imag;
	delete[] _flags;
	delete[] sqrt_weights;
}

bool DiskAndDeltaCost::Evaluate(double const* const* parameters,
                                    double* residuals,
                                    double** jacobians) const
{
	double flux    = parameters[0][0];
	double x0      = parameters[1][0];
	double y0      = parameters[2][0];
	double sigma   = parameters[3][0];
	double flux_ps = parameters[4][0];

	double size;
	double dsize_dsigma;
	double uvdist;
	double pos_real;
	double pos_imag;
	double V_mod_real;
	double V_mod_imag;

	for(int uvrow = 0; uvrow < _nrow; uvrow++)
	{
		for(int chan = 0; chan < _nchan; chan++)
		{
			for(int pol = 0; pol < _nstokes; pol++)
			{
				int index = chan+pol*_nchan+uvrow*_nstokes*_nchan;
				double& u = _u[index];
				double& v = _v[index];
				double& sqrt_weight = sqrt_weights[pol+uvrow*_nstokes];

	// 			size         = exp(-2*M_PI*M_PI*(u*u+v*v)*sigma*sigma);
				uvdist       =sqrt(u*u+v*v);
				if(sigma*uvdist < 1e-12)
				{
					size         = 0.5;
					dsize_dsigma = 0.;
				}
				else
				{
					size         = 1/M_PI*j1(2*M_PI*sigma*uvdist)/sigma/uvdist;
					dsize_dsigma = 2/M_PI*(M_PI*j0(2*M_PI*sigma*uvdist)/sigma -
										j1(2*M_PI*sigma*uvdist)/sigma/sigma/uvdist);
				}
	// 			std::cout << "dsize_dsigma = " << dsize_dsigma << std::endl;
				pos_real     = cos(-2*M_PI*(x0*u+y0*v));
				pos_imag     = sin(-2*M_PI*(x0*u+y0*v));
				V_mod_real   = flux*size*pos_real;
				V_mod_imag   = flux*size*pos_imag;

				if(_flags[index])
				{
					residuals[2*index+0] = 0.;
					residuals[2*index+1] = 0.;

					if(jacobians != NULL)
					{
						if(jacobians[0] != NULL)
						{
							jacobians[0][2*index+0] = 0.;
							jacobians[0][2*index+1] = 0.;
						}
						if(jacobians[1] != NULL)
						{
							jacobians[1][2*index+0] = 0.;
							jacobians[1][2*index+1] = 0.;
						}
						if(jacobians[2] != NULL)
						{
							jacobians[2][2*index+0] = 0.;
							jacobians[2][2*index+1] = 0.;
						}
						if(jacobians[3] != NULL)
						{
							jacobians[3][2*index+0] = 0.;
							jacobians[3][2*index+1] = 0.;
						}
						if(jacobians[4] != NULL)
						{
							jacobians[4][2*index+0] = 0.;
							jacobians[4][2*index+1] = 0.;
						}
					}
				}
				else
				{
					residuals[2*index+0] = sqrt_weight*(_V_real[index] - V_mod_real-flux_ps*pos_real);
					residuals[2*index+1] = sqrt_weight*(_V_imag[index] - V_mod_imag-flux_ps*pos_imag);

					if(jacobians != NULL)
					{
						if(jacobians[0] != NULL)
						{
							jacobians[0][2*index+0] = -sqrt_weight*size*pos_real;
							jacobians[0][2*index+1] = -sqrt_weight*size*pos_imag;
						}
						if(jacobians[1] != NULL)
						{
							jacobians[1][2*index+0] = -2*M_PI*sqrt_weight*u*V_mod_imag;
							jacobians[1][2*index+1] = 2*M_PI *sqrt_weight*u*V_mod_real;
						}
						if(jacobians[2] != NULL)
						{
							jacobians[2][2*index+0] = -2*M_PI*sqrt_weight*v*V_mod_imag;
							jacobians[2][2*index+1] = 2*M_PI *sqrt_weight*v*V_mod_real;
						}
						if(jacobians[3] != NULL)
						{
							jacobians[3][2*index+0] = -sqrt_weight*flux*dsize_dsigma*pos_real;
							jacobians[3][2*index+1] = -sqrt_weight*flux*dsize_dsigma*pos_imag;
						}
						if(jacobians[4] != NULL)
						{
							jacobians[4][2*index+0] = -sqrt_weight*pos_real;
							jacobians[4][2*index+1] = -sqrt_weight*pos_imag;
						}
					}
				}
			}
		}
	}

	return true;
}
