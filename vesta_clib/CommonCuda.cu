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
#include "CommonCuda.h"
#include "cuda_error.h"
#include "Chunk.h"

const float C_LIGHT = 299792458;
__global__ void sqrt_weights(float* weights, const int nchan, const int nstokes, const int nrow);
__global__ void cu_gaussian_size(double sigma, double x0, double y0,
                                 const int nchan, const int nstokes, const int nrow,
                                 double* u, double* v,
                                 double* size);
__global__ void cu_evaluate_gaussian(float flux, float sigma, float x0, float y0,
                             const int nchan, const int nstokes, const int nrow,
					         const VisDataContainer uvdata,
							 float* residuals, float* jacobians);
__global__ void cu_disk_size(double sigma, double x0, double y0,
                             const int nchan, const int nstokes, const int nrow,
                             double* u, double* v,
                             double* size, double* dsize_dsigma);
__global__ void cu_pos(double sigma, double x0, double y0,
                       const int nchan, const int nstokes, const int nrow,
                       double* u, double* v,
                       double* pos_real, double* pos_imag);

void allocate_stuff(const int nchan, const int nstokes, const int nrow, DataContainer& data)
{
	CudaSafeCall(cudaMalloc( (void**)&data.u, sizeof(double)*nchan*nstokes*nrow));
	CudaSafeCall(cudaMalloc( (void**)&data.v, sizeof(double)*nchan*nstokes*nrow));
	CudaSafeCall(cudaMalloc( (void**)&data.size, sizeof(double)*nchan*nstokes*nrow));
	CudaSafeCall(cudaMalloc( (void**)&data.dsize_dsigma, sizeof(double)*nchan*nstokes*nrow));
	CudaSafeCall(cudaMalloc( (void**)&data.pos_real, sizeof(double)*nchan*nstokes*nrow));
	CudaSafeCall(cudaMalloc( (void**)&data.pos_imag, sizeof(double)*nchan*nstokes*nrow));
}


// Allocate space for uv data on device and copy data over.
void setup_uvdata(Chunk& chunk, VisDataContainer& dev_uvdata)
{
	dev_uvdata.nchan = chunk.nChan();
	dev_uvdata.nstokes = chunk.nStokes();
	dev_uvdata.nrow = chunk.size();


	size_t size = sizeof(float)*dev_uvdata.nrow*dev_uvdata.nstokes*dev_uvdata.nchan;
	CudaSafeCall(cudaMalloc((void**)&dev_uvdata.u, size));
	CudaSafeCall(cudaMalloc((void**)&dev_uvdata.v, size));
	CudaSafeCall(cudaMalloc((void**)&dev_uvdata.w, size));
	CudaSafeCall(cudaMalloc((void**)&dev_uvdata.sqrt_weights, size));
	CudaSafeCall(cudaMalloc((void**)&dev_uvdata.V_real, size));
	CudaSafeCall(cudaMalloc((void**)&dev_uvdata.V_imag, size));

	// Copy u, v and w to device.
	float* u = new float[dev_uvdata.nrow*dev_uvdata.nstokes*dev_uvdata.nchan];
	float* v = new float[dev_uvdata.nrow*dev_uvdata.nstokes*dev_uvdata.nchan];
	float* w = new float[dev_uvdata.nrow*dev_uvdata.nstokes*dev_uvdata.nchan];
	for(int uvrow = 0; uvrow < chunk.size(); uvrow++)
	{
		Visibility& inVis = chunk.inVis[uvrow];
		float* freq = inVis.freq;

		for(int chan = 0; chan < chunk.nChan(); chan++)
		{
			for(int pol = 0; pol < chunk.nStokes(); pol++)
			{
				size_t index = uvrow*chunk.nChan()*chunk.nStokes()+pol*chunk.nChan()+chan;
				u[index] = inVis.u * freq[chan] / C_LIGHT;
				v[index] = inVis.v * freq[chan] / C_LIGHT;
				w[index] = inVis.w * freq[chan] / C_LIGHT;
			}
		}
	}
	CudaSafeCall(cudaMemcpy(dev_uvdata.u, u, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_uvdata.v, v, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_uvdata.w, w, size, cudaMemcpyHostToDevice));
	delete[] u;
	delete[] v;
	delete[] w;


	CudaSafeCall(cudaMemcpy(dev_uvdata.sqrt_weights, chunk.weight_in, size, cudaMemcpyHostToDevice));
	dim3 dimBlock(dev_uvdata.nchan, dev_uvdata.nstokes);
	dim3 dimGrid(dev_uvdata.nrow);
// 	sqrt_weights<<<dimGrid, dimBlock>>>(dev_uvdata.sqrt_weights,
// 	                                    dev_uvdata.nchan, dev_uvdata.nstokes, dev_uvdata.nrow);
	CudaSafeCall(cudaMemcpy(dev_uvdata.V_real, chunk.data_real_in, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_uvdata.V_imag, chunk.data_imag_in, size, cudaMemcpyHostToDevice));

	size = sizeof(int)*dev_uvdata.nrow*dev_uvdata.nstokes*dev_uvdata.nchan;
	CudaSafeCall(cudaMalloc((void**)&dev_uvdata.flag, size));
	CudaSafeCall(cudaMemcpy(dev_uvdata.flag, chunk.data_flag_in, size, cudaMemcpyHostToDevice));
}

void free_uvdata(VisDataContainer& dev_uvdata)
{
	CudaSafeCall(cudaFree(dev_uvdata.u));
	CudaSafeCall(cudaFree(dev_uvdata.v));
	CudaSafeCall(cudaFree(dev_uvdata.w));
	CudaSafeCall(cudaFree(dev_uvdata.sqrt_weights));
	CudaSafeCall(cudaFree(dev_uvdata.V_real));
	CudaSafeCall(cudaFree(dev_uvdata.V_imag));
	CudaSafeCall(cudaFree(dev_uvdata.flag));
}

__global__ void sqrt_weights(float* weights, const int nchan, const int nstokes, const int nrow)
{
	size_t chan = threadIdx.x;
	size_t pol = threadIdx.y;
	size_t row = blockIdx.x;

	if(chan < nchan and pol < nstokes and row < nrow)
	{
		size_t index = chan+pol*nchan+row*nchan*nstokes;
		if(weights[index] != 0)
			weights[index] = sqrt(weights[index]);
	}
}

void free_stuff(DataContainer& data)
{
	CudaSafeCall(cudaFree( data.u));
	CudaSafeCall(cudaFree( data.v));
	CudaSafeCall(cudaFree( data.size));
	CudaSafeCall(cudaFree( data.dsize_dsigma));
	CudaSafeCall(cudaFree( data.pos_real));
	CudaSafeCall(cudaFree( data.pos_imag));
}

void evaluate_gaussian(float flux, float sigma, float x0, float y0,
                       const int nchan, const int nstokes, const int nrow,
					   const VisDataContainer uvdata,
					   float* residuals, float** jacobians)
{
	dim3 dimBlock(uvdata.nchan, uvdata.nstokes);
	dim3 dimGrid(uvdata.nrow);
// 	dim3 dimBlock(nchan, nstokes);
// 	dim3 dimGrid(nrow);
	
	// setup storage for resiudals and jacobians
	float* dev_residuals;
	float* dev_jacobians;

// 	int nres = nchan*nstokes*nrow*2;
	int nres = uvdata.nchan*uvdata.nstokes*uvdata.nrow*2;
	CudaSafeCall(cudaMalloc((void**)&dev_residuals, sizeof(float)*nres));
	if(jacobians != NULL)
		CudaSafeCall(cudaMalloc((void**)&dev_jacobians, sizeof(float)*nres*4));
	else
		dev_jacobians = NULL;

	cu_evaluate_gaussian<<<dimGrid, dimBlock>>>(flux, sigma, x0, y0,
			                                    uvdata.nchan, uvdata.nstokes, uvdata.nrow,
												uvdata, dev_residuals, dev_jacobians);

	CudaSafeCall(cudaMemcpy(residuals, dev_residuals, sizeof(float)*nres, cudaMemcpyDeviceToHost));
	if(jacobians != NULL)
	{
		if(jacobians[0] != NULL)
		{
			CudaSafeCall(cudaMemcpy(jacobians[0], &dev_jacobians[nres*0],
						sizeof(float)*nres, cudaMemcpyDeviceToHost));
		}
		if(jacobians[1] != NULL)
		{
			CudaSafeCall(cudaMemcpy(jacobians[1], &dev_jacobians[nres*1],
						sizeof(float)*nres, cudaMemcpyDeviceToHost));
		}
		if(jacobians[2] != NULL)
		{
			CudaSafeCall(cudaMemcpy(jacobians[2], &dev_jacobians[nres*2],
						sizeof(float)*nres, cudaMemcpyDeviceToHost));
		}
		if(jacobians[3] != NULL)
		{
			CudaSafeCall(cudaMemcpy(jacobians[3], &dev_jacobians[nres*3],
						sizeof(float)*nres, cudaMemcpyDeviceToHost));
		}
	}
	CudaSafeCall(cudaFree(dev_residuals));
	if(dev_jacobians != NULL)
		CudaSafeCall(cudaFree(dev_jacobians));
}

void calc_gaussian(double sigma, double x0, double y0,
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


void calc_disk(double sigma, double x0, double y0,
               const int nchan, const int nstokes, const int nrow,
               double* u, double* v,
               double* size, double* dsize_dsigma, double* pos_real, double* pos_imag,
			   const DataContainer data)
{
	CudaSafeCall(cudaMemcpy( data.u, u, sizeof(double)*nchan*nstokes*nrow, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy( data.v, v, sizeof(double)*nchan*nstokes*nrow, cudaMemcpyHostToDevice));
	dim3 dimBlock(nchan, nstokes);
	dim3 dimGrid(nrow);
	cu_disk_size<<<dimGrid, dimBlock>>>(sigma, x0, y0, nchan, nstokes, nrow,
	                                        data.u, data.v, data.size, data.dsize_dsigma);
	cu_pos<<<dimGrid, dimBlock>>>(sigma, x0, y0, nchan, nstokes, nrow,
	                              data.u, data.v, data.pos_real, data.pos_imag);

	CudaSafeCall(cudaMemcpy(size, data.size, sizeof(double)*nchan*nstokes*nrow, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(dsize_dsigma, data.dsize_dsigma, sizeof(double)*nchan*nstokes*nrow, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(pos_real, data.pos_real, sizeof(double)*nchan*nstokes*nrow, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(pos_imag, data.pos_imag, sizeof(double)*nchan*nstokes*nrow, cudaMemcpyDeviceToHost));
}

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
}

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
}


__global__ void cu_evaluate_gaussian(float flux, float sigma, float x0, float y0,
                             const int nchan, const int nstokes, const int nrow,
					         const VisDataContainer uvdata,
							 float* residuals, float* jacobians)
{
	size_t chan = threadIdx.x;
	size_t pol = threadIdx.y;
	size_t row = blockIdx.x;

	if(chan < nchan and pol < nstokes and row < nrow)
	{
		size_t index = chan+pol*nchan+row*nchan*nstokes;

		float& u = uvdata.u[index];
		float& v = uvdata.v[index];
		float& sqrt_weight = uvdata.sqrt_weights[pol+row*nstokes*nchan];

		float size     = exp(-2*M_PI*M_PI*(u*u+v*v)*sigma*sigma);
		float pos_real = cos(-2*M_PI*(x0*u+y0*v));
		float pos_imag = sin(-2*M_PI*(x0*u+y0*v));
		float V_mod_real = flux*size*pos_real;
		float V_mod_imag = flux*size*pos_imag;

		if(uvdata.flag[index])
		{
// 			residuals[2*index+0] = index;
// 			residuals[2*index+1] = index;
			residuals[2*index+0] = 0.0;
			residuals[2*index+1] = 0.0;
			if(jacobians != NULL)
			{
				int nres = 2*nchan*nstokes*nrow;
				jacobians[0*nres+2*index+0] = 0.;
				jacobians[0*nres+2*index+1] = 0.;
				jacobians[1*nres+2*index+0] = 0.;
				jacobians[1*nres+2*index+1] = 0.;
				jacobians[2*nres+2*index+0] = 0.;
				jacobians[2*nres+2*index+1] = 0.;
				jacobians[3*nres+2*index+0] = 0.;
				jacobians[3*nres+2*index+1] = 0.;
			}
		}
		else
		{
// 			residuals[2*index+0] = 1.;
// 			residuals[2*index+1] = 1.;
// 			residuals[2*index+0] = flux;
// 			residuals[2*index+1] = flux;
// 			residuals[2*index+0] = V_mod_real;
// 			residuals[2*index+1] = V_mod_imag;
// 			residuals[2*index+0] = (uvdata.V_real[index]-V_mod_real);
// 			residuals[2*index+1] = (uvdata.V_imag[index]-V_mod_imag);
// 			residuals[2*index+0] = index;
// 			residuals[2*index+1] = index;
			residuals[2*index+0] = sqrt_weight*(uvdata.V_real[index]-V_mod_real);
			residuals[2*index+1] = sqrt_weight*(uvdata.V_imag[index]-V_mod_imag);
			if(jacobians != NULL)
			{
				int nres = 2*nchan*nstokes*nrow;
				jacobians[0*nres+2*index+0] = -sqrt_weight*size*pos_real; // dchi/dflux
				jacobians[0*nres+2*index+1] = -sqrt_weight*size*pos_imag;
				jacobians[1*nres+2*index+0] = -2*M_PI*sqrt_weight*u*V_mod_imag; // dchi/u
				jacobians[1*nres+2*index+1] = 2*M_PI *sqrt_weight*u*V_mod_real;
				jacobians[2*nres+2*index+0] = -2*M_PI*sqrt_weight*v*V_mod_imag; // dchi/v
				jacobians[2*nres+2*index+1] = 2*M_PI *sqrt_weight*v*V_mod_real;
				jacobians[3*nres+2*index+0] = sqrt_weight*V_mod_real * 2*M_PI*M_PI * 2*sigma*(u*u+v*v); // dchi/sigma
				jacobians[3*nres+2*index+1] = sqrt_weight*V_mod_imag * 2*M_PI*M_PI * 2*sigma*(u*u+v*v);
			}
		}
	}
}
__global__ void cu_disk_size(double sigma, double x0, double y0,
                             const int nchan, const int nstokes, const int nrow,
                             double* u, double* v,
                             double* size, double* dsize_dsigma)
{
	size_t chan = threadIdx.x;
	size_t pol = threadIdx.y;
	size_t row = blockIdx.x;

	if(chan < nchan and pol < nstokes and row < nrow)
	{
		size_t index = chan+pol*nchan+row*nchan*nstokes;
		double uvdist = sqrt(u[index]*u[index]+v[index]*v[index]);
		if(sigma*uvdist < 1e-12)
		{
			size[index] = .5;
			dsize_dsigma[index] = 0.;
		}
		else
		{
			size[index] = 1/M_PI*j1(2*M_PI*sigma*uvdist)/sigma/uvdist;
			dsize_dsigma[index] = 2/M_PI*(M_PI*j0(2*M_PI*sigma*uvdist)/sigma -
					j1(2*M_PI*sigma*uvdist)/sigma/sigma/uvdist);
		}
	}
}
