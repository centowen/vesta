// vesta, use Ceres to do uv-model fitting for ms-data.
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
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <queue>
#include <ceres/ceres.h>
#include <pthread.h>

#include "ndarray.h"
#include "problem_setup.h"
#include "msio.h"
#include "DiskCost.h"
#include "DiskAndDeltaCost.h"
#include "PointSourceCostFunction.h"
#include "GaussianCostFunctionCircular.h"
#include "GaussianCostFunctionCircularAndPointSource.h"

using std::cout;
using std::endl;
using std::queue;
using std::pair;
using ceres::CostFunction;

typedef std::vector<CostFunction*> Costs;

class ComputerDataContainer
{
public:
	Costs& costs;
	double& chi2;

	int n_pars;
	double** parameters;

	ComputerDataContainer(Costs& costs_, double& chi2_, Ndarray<double, 1> parameters);
	~ComputerDataContainer();
};


template <int ndim>
void chi2_scan(Ndarray<double, ndim> chi2, Costs& costs,
               Ndarray<double, ndim+1> parameters,
               vector<pthread_t>& threads);

template <>
void chi2_scan<1>(Ndarray<double, 1> chi2, Costs& costs,
                  Ndarray<double, 2> parameters,
                  vector<pthread_t>& threads);


Costs get_cost_functions(std::string path, int model);
void* computer_thread(void* data);


extern "C"{
void c_chi2_scan(numpyArray<double> c_chi2, int ndim,
				 const char* c_path,
				 int model, numpyArray<double> c_parameters)
{
	std::string path(c_path);
	Costs costs = get_cost_functions(path, model);
	cout << "model is set up " << (**costs.begin()).num_residuals() << endl;
	vector<pthread_t> threads;

	if(ndim == 1)
	{
		Ndarray<double, 1> chi2(c_chi2);
		Ndarray<double, 2> parameters(c_parameters);
		chi2_scan(chi2, costs, parameters, threads);
	}
	else if(ndim == 2)
	{
		Ndarray<double, 2> chi2(c_chi2);
		Ndarray<double, 3> parameters(c_parameters);
		chi2_scan(chi2, costs, parameters, threads);
	}
	else if(ndim == 3)
	{
		Ndarray<double, 3> chi2(c_chi2);
		Ndarray<double, 4> parameters(c_parameters);
		chi2_scan(chi2, costs, parameters, threads);
	}
	else if(ndim == 4)
	{
		Ndarray<double, 4> chi2(c_chi2);
		Ndarray<double, 5> parameters(c_parameters);
		chi2_scan(chi2, costs, parameters, threads);
	}

	for(vector<pthread_t>::iterator it = threads.begin();
		it != threads.end(); it++)
	{
		pthread_join(*it, NULL);
	}

	for(Costs::iterator it = costs.begin(); it != costs.end(); it++)
	{
		delete *it;
		*it = NULL;
	}
};
};

template <int ndim>
void chi2_scan(Ndarray<double, ndim> chi2, Costs& costs,
               Ndarray<double, ndim+1> parameters,
               vector<pthread_t>& threads)
{
	for(int i = 0; i < chi2.getShape(0); i++)
		chi2_scan<ndim-1>(chi2[i], costs, parameters[i], threads);
}

template <>
void chi2_scan<1>(Ndarray<double, 1> chi2, Costs& costs,
                  Ndarray<double, 2> parameters, vector<pthread_t>& threads)
{
	for(int i = 0; i < chi2.getShape(0); i++)
	{
		chi2[i] = 0.;

		ComputerDataContainer *data = new ComputerDataContainer(costs, chi2[i], parameters[i]);
		pthread_t thread;
		threads.push_back(thread);
		pthread_create(&thread, NULL, computer_thread, (void*)data);
	}
}


std::vector<CostFunction*> get_cost_functions(std::string path, int model)
{
	Chunk chunk(chunk_size);
	DataIO* data = new msio(path.c_str(), "", msio::col_corrected_data, true, "0");
	std::vector<CostFunction*> cost_functions;


	while(data->readChunk(chunk))
	{
		float* u = new float[chunk.nChan() * chunk.nStokes()];
		float* v = new float[chunk.nChan() * chunk.nStokes()];

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

			CostFunction* cost_function;
			if(model == mod_gaussian)
			{
				cost_function = 
					new GaussianCostFunctionCircular(u, v, 
													inVis.data_real, inVis.data_imag,
													inVis.weight, inVis.data_flag,
													chunk.nChan(), chunk.nStokes());
			}
			else if(model == mod_gaussian_ps)
			{
				cost_function = 
					new GaussianCostFunctionCircularAndPointSource(u, v, 
													inVis.data_real, inVis.data_imag,
													inVis.weight, inVis.data_flag,
													chunk.nChan(), chunk.nStokes());
			}
			else if(model == mod_ps)
			{
				cost_function = 
					new PointSourceCostFunction(u, v, 
					                            inVis.data_real, inVis.data_imag,
					                            inVis.weight, inVis.data_flag,
					                            chunk.nChan(), chunk.nStokes());
			}
			else if(model == mod_disk)
			{
				cost_function = 
					new DiskCost(u, v, inVis.data_real, inVis.data_imag,
								inVis.weight, inVis.data_flag,
								chunk.nChan(), chunk.nStokes());
			}
			else if(model == mod_disk_ps)
			{
				cost_function = 
					new DiskAndDeltaCost(u, v, inVis.data_real, inVis.data_imag,
										inVis.weight, inVis.data_flag,
										chunk.nChan(), chunk.nStokes());
			}

			cost_functions.push_back(cost_function);
		}

		delete[] u;
		delete[] v;
	}

	delete data;
	return cost_functions;
}

void* computer_thread(void* data)
{
	ComputerDataContainer* computer_data_container = (ComputerDataContainer*)data;

	Costs& costs = computer_data_container->costs;
	double& chi2 = computer_data_container->chi2;

	double** parameters = computer_data_container->parameters;
	int n_pars = computer_data_container->n_pars;


	Costs::iterator begin, end;
	int n_res = -1;
	double* residuals = NULL;

	for(Costs::iterator it = costs.begin(); it != costs.end(); it++)
	{
		if(n_res != (**it).num_residuals())
		{
			if(residuals != NULL)
				delete[] residuals;
			n_res = (**it).num_residuals();
			residuals = new double[n_res];
		}

		(**it).Evaluate(parameters, residuals, NULL);

		for(int i_res = 0; i_res < n_res; i_res++)
		{
			chi2 += residuals[i_res]*residuals[i_res];
		}
	}

	if(residuals != NULL)
	{
		delete[] residuals;
		residuals = NULL;
	}
}


ComputerDataContainer::ComputerDataContainer(Costs& costs_, double& chi2_,
                                             Ndarray<double, 1> parameters_): 
		costs(costs_), chi2(chi2_)
{
	n_pars = parameters_.getShape(0);
	parameters = new double*[n_pars];

	for(int i = 0; i < n_pars; i++)
	{
		parameters[i] = new double[1];
		parameters[i][0] = parameters_[i];
	}
};

ComputerDataContainer::~ComputerDataContainer()
{
	for(int i = 0; i < n_pars; i++)
	{
		delete[] parameters[i];
	}
	delete[] parameters;
};
