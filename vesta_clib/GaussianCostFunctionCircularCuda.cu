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
#include "GaussianCostFunctionCircularCuda.h"

__global__ void cu_gaussian_size(double sigma, double x0, double y0,
                                 const int nchan, const int nstokes, const int nrow,
                                 double* u, double* v,
                                 double* size)
{
	size_t chan = threadIdx.x;
	size_t pol = threadIdx.y;
	size_t row = blockIdx.x;

	if(chan < nchan and pol < nstokes and row < nrow)
	{
		size_t index = chan+pol*nchan+row*nchan*nstokes;
		size[index] = exp(-2*M_PI*M_PI*(u[index]*u[index]+v[index]*v[index])*sigma*sigma);
	}
};

__global__ void cu_pos(double sigma, double x0, double y0,
                       const int nchan, const int nstokes, const int nrow,
                       double* u, double* v,
                       double* pos_real, double* pos_imag)
{
	size_t chan = threadIdx.x;
	size_t pol = threadIdx.y;
	size_t row = blockIdx.x;

	if(chan < nchan and pol < nstokes and row < nrow)
	{
		size_t index = chan+pol*nchan+row*nchan*nstokes;
		pos_real[index] = cos(-2*M_PI*(x0*u[index]+y0*v[index]));
		pos_imag[index] = sin(-2*M_PI*(x0*u[index]+y0*v[index]));
	}
};

void calc_functions(double sigma, double x0, double y0,
                    const int nchan, const int nstokes, const int nrow,
                    double* u, double* v,
                    double* size, double* pos_real, double* pos_imag,
					const DataContainer data)
{
	CudaSafeCall(cudaMemcpy( data.u, u, sizeof(double)*nchan*nstokes*nrow, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy( data.v, v, sizeof(double)*nchan*nstokes*nrow, cudaMemcpyHostToDevice));
	dim3 dimBlock(nchan, nstokes);
	dim3 dimGrid(nrow);
	cu_gaussian_size<<<dimGrid, dimBlock>>>(sigma, x0, y0, nchan, nstokes, nrow,
	                                        data.u, data.v, data.size);
	cu_pos<<<dimGrid, dimBlock>>>(sigma, x0, y0, nchan, nstokes, nrow,
	                              data.u, data.v, data.pos_real, data.pos_imag);

	CudaSafeCall(cudaMemcpy(size, data.size, sizeof(double)*nchan*nstokes*nrow, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(pos_real, data.pos_real, sizeof(double)*nchan*nstokes*nrow, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(pos_imag, data.pos_imag, sizeof(double)*nchan*nstokes*nrow, cudaMemcpyDeviceToHost));
}

void allocate_stuff(const int nchan, const int nstokes, const int nrow, DataContainer& data)
{
	CudaSafeCall(cudaMalloc( (void**)&data.u, sizeof(double)*nchan*nstokes*nrow));
	CudaSafeCall(cudaMalloc( (void**)&data.v, sizeof(double)*nchan*nstokes*nrow));
	CudaSafeCall(cudaMalloc( (void**)&data.size, sizeof(double)*nchan*nstokes*nrow));
	CudaSafeCall(cudaMalloc( (void**)&data.pos_real, sizeof(double)*nchan*nstokes*nrow));
	CudaSafeCall(cudaMalloc( (void**)&data.pos_imag, sizeof(double)*nchan*nstokes*nrow));
}

void free_stuff(DataContainer& data)
{
	CudaSafeCall(cudaFree( data.u));
	CudaSafeCall(cudaFree( data.v));
	CudaSafeCall(cudaFree( data.size));
	CudaSafeCall(cudaFree( data.pos_real));
	CudaSafeCall(cudaFree( data.pos_imag));
}
