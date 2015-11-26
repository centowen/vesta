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
#include <iostream>
#include "cuda_error.h"

double* dev_u;
double* dev_v;
double* dev_size;
double* dev_pos_real;
double* dev_pos_imag;

__global__ void cuEvaluate(double sigma, double x0, double y0, int nchan, int nstokes,
                           double* u, double* v,
                           double* size, double* pos_real, double* pos_imag)
{
	size_t chan = threadIdx.x;
	size_t pol = blockIdx.x;

	if(chan < nchan and pol < nstokes)
	{
		size_t index = chan+pol*nchan;
		size[index] = exp(-2*M_PI*M_PI*(u[index]*u[index]+v[index]*v[index])*sigma*sigma);
		pos_real[index] = cos(-2*M_PI*(x0*u[index]+y0*v[index]));
		pos_imag[index] = sin(-2*M_PI*(x0*u[index]+y0*v[index]));
	}
};

void calc_functions(double sigma, double x0, double y0, int nchan, int nstokes,
                    double* u, double* v,
					double* size, double* pos_real, double* pos_imag)
{
	CudaSafeCall(cudaMemcpy( dev_u, u, sizeof(double)*nchan*nstokes, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy( dev_v, v, sizeof(double)*nchan*nstokes, cudaMemcpyHostToDevice));
	cuEvaluate<<<nchan, nstokes>>>(sigma, x0, y0, nchan, nstokes, dev_u, dev_v, dev_size, dev_pos_real, dev_pos_imag);
	CudaSafeCall(cudaMemcpy(size, dev_size, sizeof(double)*nchan*nstokes, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(pos_real, dev_pos_real, sizeof(double)*nchan*nstokes, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(pos_imag, dev_pos_imag, sizeof(double)*nchan*nstokes, cudaMemcpyDeviceToHost));
}

void allocate_stuff(const int nchan, const int nstokes)
{
	CudaSafeCall(cudaMalloc( (void**)&dev_u, sizeof(double)*nchan*nstokes));
	CudaSafeCall(cudaMalloc( (void**)&dev_v, sizeof(double)*nchan*nstokes));
	CudaSafeCall(cudaMalloc( (void**)&dev_size, sizeof(double)*nchan*nstokes));
	CudaSafeCall(cudaMalloc( (void**)&dev_pos_real, sizeof(double)*nchan*nstokes));
	CudaSafeCall(cudaMalloc( (void**)&dev_pos_imag, sizeof(double)*nchan*nstokes));
}

void free_stuff()
{
	CudaSafeCall(cudaFree( dev_u));
	CudaSafeCall(cudaFree( dev_v));
	CudaSafeCall(cudaFree( dev_size));
	CudaSafeCall(cudaFree( dev_pos_real));
	CudaSafeCall(cudaFree( dev_pos_imag));
}
