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
#include "msio.h"
#include "problem_setup.h"
#include "GaussianCostFunctionCircular.h"

// using ceres::Problem;
// using std::string;
using std::cout;
using std::endl;

void add_residual_blocks(Problem& problem, string path, 
                         double& flux, double& sigma, double& x0, double& y0)
{
	Chunk chunk(chunk_size);
	DataIO* data = new msio(path.c_str(), "", msio::col_corrected_data);
	while(data->readChunk(chunk))
	{
		add_chunk_to_residual_blocks(problem, chunk, flux, sigma, x0, y0);
	}
	delete data;
}

void add_chunk_to_residual_blocks(Problem& problem, Chunk& chunk,
                                  double& flux, double& sigma,
								  double& x0, double& y0)
{
	float* u = new float[chunk.nChan() * chunk.nStokes()];
	float* v = new float[chunk.nChan() * chunk.nStokes()];

	flux = 1e-3;
	sigma = M_PI/180./3600;
	x0 = 0.;
	y0 = 0.;

	for(int uvrow = 0; uvrow < chunk.size(); uvrow++)
	{
		Visibility& inVis = chunk.inVis[uvrow];
		float* freq = inVis.freq;

		for(int chan = 0; chan < chunk.nChan(); chan++)
		{
			for(int i_stokes = 0; i_stokes < chunk.nStokes(); i_stokes++)
			{
				size_t index = i_stokes*chunk.nChan()+chan;
				u[index] = inVis.u * freq[chan] / C_LIGHT;
				v[index] = inVis.v * freq[chan] / C_LIGHT;
			}
		}
		CostFunction* cost_function = 
			new GaussianCostFunctionCircular(u, v, 
			                                 inVis.data_real, inVis.data_imag,
			                                 inVis.weight, NULL,
			                                 chunk.nChan(), chunk.nStokes());
		problem.AddResidualBlock(cost_function, NULL, &flux, &x0, &y0, &sigma);
		problem.SetParameterBlockConstant(&x0);
		problem.SetParameterBlockConstant(&y0);
		problem.SetParameterLowerBound(&flux, 0, 0.);
		problem.SetParameterLowerBound(&sigma, 0, 0.);
		problem.SetParameterUpperBound(&sigma, 0,  M_PI/180./3600*15);

	}
	delete[] u;
	delete[] v;
}
