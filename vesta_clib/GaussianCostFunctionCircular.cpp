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
#include "GaussianCostFunctionCircular.h"
#include "Chunk.h"

const float C_LIGHT = 299792458;

GaussianCostFunctionCircular::GaussianCostFunctionCircular(Chunk chunk)
	: _nchan(chunk.nChan()), _nstokes(chunk.nStokes()), _nrow(chunk.size())
{
	for(int i = 0; i < 4; i++) {
		mutable_parameter_block_sizes()->push_back(1);
	}
	set_num_residuals(2*_nchan*_nstokes*_nrow);

#ifdef ENABLE_CUDA
	setup_uvdata(chunk, dev_uvdata);
#else
	_u = new double[_nchan*_nstokes*_nrow];
	_v = new double[_nchan*_nstokes*_nrow];
	_V_real  = new double[_nchan*_nstokes*_nrow];
	_V_imag  = new double[_nchan*_nstokes*_nrow];
	_flags   = new   bool[_nchan*_nstokes*_nrow];
	sqrt_weights = new double[_nstokes*_nrow];
	size = new double[_nchan*_nstokes*_nrow];
	pos_real = new double[_nchan*_nstokes*_nrow];
	pos_imag = new double[_nchan*_nstokes*_nrow];

	size_t index;
	for(int uvrow = 0; uvrow < _nrow; uvrow++)
	{
		for(int chan = 0; chan < _nchan; chan++)
		{
			for(int pol = 0; pol < _nstokes; pol++)
			{
				Visibility& inVis = chunk.inVis[uvrow];
				float* freq = inVis.freq;

				index = _nstokes*_nchan*uvrow+chan+pol*_nchan;
				_u[index] = inVis.u * freq[chan] / C_LIGHT;
				_v[index] = inVis.v * freq[chan] / C_LIGHT;
				_V_real[index] = (double)chunk.data_real_in[index];
				_V_imag[index] = (double)chunk.data_imag_in[index];
				if(chunk.data_flag_in != NULL)
					_flags[index]  = chunk.data_flag_in [index];

				if(chan == 0) 
				{
					sqrt_weights[_nstokes*uvrow+pol] =
						sqrt((double)inVis.weight[pol]);
				}
			}
		}
	}
#endif
}

GaussianCostFunctionCircular::~GaussianCostFunctionCircular()
{
#ifdef ENABLE_CUDA
	free_uvdata(dev_uvdata);
#else
	delete[] _u;
	delete[] _v;
	delete[] size;
	delete[] pos_real;
	delete[] pos_imag;
	delete[] _V_real;
	delete[] _V_imag;
	delete[] _flags;
	delete[] sqrt_weights;
#endif
}

bool GaussianCostFunctionCircular::Evaluate(double const* const* parameters,
                                    double* residuals,
                                    double** jacobians) const
{
	double flux  = parameters[0][0];
	double x0    = parameters[1][0];
	double y0    = parameters[2][0];
	double sigma = parameters[3][0];

#ifdef ENABLE_CUDA
	int nres = 2*_nchan*_nstokes*_nrow;
	float* residuals_f = new float[nres];
	float** jacobians_f;

	if(jacobians == NULL)
		jacobians_f = NULL;
	else
	{
		jacobians_f = new float*[4];
		for(int i = 0; i < 4; i++)
		{
			if(jacobians[i] != NULL)
				jacobians_f[i] = new float[nres];
			else
				jacobians_f[i] = NULL;
		}
	}
	evaluate_gaussian(flux, sigma, x0, y0,
	                  _nchan, _nstokes, _nrow,
					  dev_uvdata,
					  residuals_f, jacobians_f);

	for(int i = 0; i < nres; i++)
		residuals[i] = (double)residuals_f[i];
	delete[] residuals_f;

	if(jacobians != NULL)
	{
		for(int par = 0; par < 4; par++)
		{
			if(jacobians[par] != NULL)
			{
				for(int i = 0; i < nres; i++)
					jacobians[par][i] = (double)jacobians_f[par][i];
			}
			delete[] jacobians_f[par];
		}
		delete[] jacobians_f;

	}

#else
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
				size[index]     = exp(-2*M_PI*M_PI*(u*u+v*v)*sigma*sigma);
				pos_real[index] = cos(-2*M_PI*(x0*u+y0*v));
				pos_imag[index] = sin(-2*M_PI*(x0*u+y0*v));
			}
		}
	}


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

				V_mod_real   = flux*size[index]*pos_real[index];
				V_mod_imag   = flux*size[index]*pos_imag[index];

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
					}
				}
				else
				{
					residuals[2*index+0] = sqrt_weight*(_V_real[index] - V_mod_real);
					residuals[2*index+1] = sqrt_weight*(_V_imag[index] - V_mod_imag);

					if(jacobians != NULL)
					{
						if(jacobians[0] != NULL)
						{
							jacobians[0][2*index+0] = -sqrt_weight*size[index]*pos_real[index];
							jacobians[0][2*index+1] = -sqrt_weight*size[index]*pos_imag[index];
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
							jacobians[3][2*index+0] = sqrt_weight*V_mod_real * 2*M_PI*M_PI * 2*sigma*(u*u+v*v);
							jacobians[3][2*index+1] = sqrt_weight*V_mod_imag * 2*M_PI*M_PI * 2*sigma*(u*u+v*v);
						}
					}
				}
			}
		}
	}
#endif

	return true;
}
