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
#include "DiskCost.h"
#include "DiskAndDeltaCost.h"
#include "PointSourceCostFunction.h"
#include "GaussianCostFunctionCircular.h"
#include "GaussianCostFunctionCircularAndPointSource.h"

// using ceres::Problem;
// using std::string;
using std::cout;
using std::endl;

void add_residual_blocks(Problem& problem, string path, 
                         double& flux, double& sigma, double& x0, double& y0,
                         double& flux_point_source,
						 int model)
{
	Chunk chunk(100);
// 	Chunk chunk(chunk_size);
// 	DataIO* data = new msio(path.c_str(), "", msio::col_corrected_data);
	DataIO* data = new msio(path.c_str(), "", msio::col_corrected_data, true, "0");
	while(data->readChunk(chunk))
	{
		add_chunk_to_residual_blocks(problem, chunk, flux, sigma, x0, y0,
				                     flux_point_source, model);
	}
	delete data;
}

void add_chunk_to_residual_blocks(Problem& problem, Chunk& chunk,
                                  double& flux, double& sigma,
								  double& x0, double& y0,
								  double& flux_point_source,
								  int model)
{
	bool fixed_position = false;
	ceres::LossFunction* loss_function = NULL;
// 	ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);
// 	ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

	float* u = new float[chunk.nChan() * chunk.nStokes() * chunk.size()];
	float* v = new float[chunk.nChan() * chunk.nStokes() * chunk.size()];

	if(model == mod_gaussian_ps)
	{
// 		flux = 2.8e-3;
		flux = 1.4e-3;
// 		sigma = 0.426*M_PI/180./3600;
		sigma = 0.4*M_PI/180./3600;
		x0 = 0.;
		y0 = 0.;
		flux_point_source = 0.5e-3;
	}
	else if(model == mod_gaussian)
	{
// 		flux = 1.2e-3;
		flux = 4.22e-3;
// 		sigma = 0.4*M_PI/180./3600;
		sigma = 0.3*M_PI/180./3600;
		x0 = 0.;
		y0 = 0.;
	}
	else if(model == mod_ps)
	{
		flux = 0.;
		sigma = 0.;
		x0 = 0.;
		y0 = 0.;
		flux_point_source = 1.0e-3;
	}
	else if(model == mod_disk)
	{
		flux = 2.4e-3;
		sigma = 1.0*M_PI/180./3600;
		x0 = 0.;
		y0 = 0.;
	}
	else if(model == mod_disk_ps)
	{
		flux = 0.47e-3;
		sigma = 1.2*M_PI/180./3600;
		x0 = 0.;
		y0 = 0.;
		flux_point_source = 1.00e-3;
	}

	for(int uvrow = 0; uvrow < chunk.size(); uvrow++)
	{
		Visibility& inVis = chunk.inVis[uvrow];
		float* freq = inVis.freq;

		for(int chan = 0; chan < chunk.nChan(); chan++)
		{
			for(int i_stokes = 0; i_stokes < chunk.nStokes(); i_stokes++)
			{
				size_t index = uvrow*chunk.nChan()*chunk.nStokes()+i_stokes*chunk.nChan()+chan;
				u[index] = inVis.u * freq[chan] / C_LIGHT;
				v[index] = inVis.v * freq[chan] / C_LIGHT;
			}
		}
	}
// 		CostFunction* cost_function = 
// 			new PointSourceCostFunction(u, v, 
// 			                            inVis.data_real, inVis.data_imag,
// 			                            inVis.weight, inVis.data_flag,
// 			                            chunk.nChan(), chunk.nStokes());

	if(model == mod_gaussian)
	{
// 		CostFunction* cost_function = 
// 			new GaussianCostFunctionCircular(u, v, 
// 											chunk.data_real_in, chunk.data_imag_in,
// 											chunk.weight_in, chunk.data_flag_in,
// 											chunk.nChan(), chunk.nStokes(), chunk.size());
		CostFunction* cost_function = 
			new GaussianCostFunctionCircular(chunk);
// 		problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(1.0), &flux, &x0, &y0, &sigma);
		problem.AddResidualBlock(cost_function, loss_function, &flux, &x0, &y0, &sigma);
// 		problem.AddResidualBlock(cost_function, NULL, &flux, &x0, &y0);
		problem.SetParameterLowerBound(&flux, 0, 0.);
		problem.SetParameterLowerBound(&sigma, 0, 0.);
// 			problem.SetParameterLowerBound(&sigma, 0,  -M_PI/180./3600*15);
		problem.SetParameterUpperBound(&sigma, 0,  M_PI/180./3600*15);
// 			problem.SetParameterBlockConstant(&sigma);
	}
	else if(model == mod_gaussian_ps)
	{
		CostFunction* cost_function = 
			new GaussianCostFunctionCircularAndPointSource(u, v, 
											chunk.data_real_in, chunk.data_imag_in,
											chunk.weight_in, chunk.data_flag_in,
											chunk.nChan(), chunk.nStokes(), chunk.size());
		problem.AddResidualBlock(cost_function, loss_function, &flux, &x0, &y0, &sigma, &flux_point_source);
		problem.SetParameterLowerBound(&flux, 0, 0.);
		problem.SetParameterLowerBound(&sigma, 0, 0.);
// 			problem.SetParameterUpperBound(&sigma, 0,  M_PI/180./3600*5);
// 			problem.SetParameterUpperBound(&sigma, 0,  M_PI/180./3600*0.54);
		problem.SetParameterUpperBound(&sigma, 0,  M_PI/180./3600*0.64);
// 			problem.SetParameterUpperBound(&sigma, 0,  M_PI/180./3600*2.56/4.*3./2.35);
// 			problem.SetParameterBlockConstant(&sigma);
		problem.SetParameterLowerBound(&flux_point_source, 0, 0.);
	}
	else if(model == mod_ps)
	{
		CostFunction* cost_function = 
			new PointSourceCostFunction(u, v, 
					chunk.data_real_in, chunk.data_imag_in,
					chunk.weight_in, chunk.data_flag_in,
					chunk.nChan(), chunk.nStokes(), chunk.size());
		problem.AddResidualBlock(cost_function, loss_function, &flux_point_source, &x0, &y0);
		problem.SetParameterLowerBound(&flux_point_source, 0, 0.);
	}
	else if(model == mod_disk)
	{
		CostFunction* cost_function = 
			new DiskCost(u, v,
					chunk.data_real_in, chunk.data_imag_in,
					chunk.weight_in, chunk.data_flag_in,
					chunk.nChan(), chunk.nStokes(), chunk.size());
		problem.AddResidualBlock(cost_function, loss_function, &flux, &x0, &y0, &sigma);
		problem.SetParameterLowerBound(&flux, 0, 0.);
		problem.SetParameterLowerBound(&sigma, 0, 0.);
		problem.SetParameterUpperBound(&sigma, 0,  M_PI/180./3600*15);
	}
	else if(model == mod_disk_ps)
	{
		CostFunction* cost_function = 
			new DiskAndDeltaCost(u, v,
					chunk.data_real_in, chunk.data_imag_in,
					chunk.weight_in, chunk.data_flag_in,
					chunk.nChan(), chunk.nStokes(), chunk.size());
		problem.AddResidualBlock(cost_function, loss_function, &flux, &x0, &y0, &sigma, &flux_point_source);
		problem.SetParameterLowerBound(&flux, 0, 0.);
		problem.SetParameterLowerBound(&sigma, 0, 0.);
		problem.SetParameterUpperBound(&sigma, 0,  M_PI/180./3600*15);
		problem.SetParameterLowerBound(&flux_point_source, 0, 0.);
	}
	if(fixed_position)
	{
		problem.SetParameterBlockConstant(&x0);
		problem.SetParameterBlockConstant(&y0);
	}

	delete[] u;
	delete[] v;
}
