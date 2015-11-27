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

#ifndef __COMMON_CUDA_H__
#define __COMMON_CUDA_H__

class Chunk;
class DataContainer
{
public:
	double* u;
	double* v;
	double* size;
	double* dsize_dsigma;
	double* pos_real;
	double* pos_imag;
};

class VisDataContainer
{
public:
	size_t nchan;
	size_t nstokes;
	size_t nrow;

	float* u;
	float* v;
	float* w;

	float* sqrt_weights;
	float* V_real;
	float* V_imag;
	int* flag;
};



void allocate_stuff(const int nchan, const int nstokes, const int nrow,
                    DataContainer& data);
void setup_uvdata(Chunk& chunk, VisDataContainer& dev_uvdata);
void free_uvdata(VisDataContainer& dev_uvdata);
void free_stuff(DataContainer& data);
void evaluate_gaussian(float flux, float sigma, float x0, float y0,
                       const int nchan, const int nstokes, const int nrow,
					   const VisDataContainer uvdata,
					   float* residual, float** jacobians);
void calc_gaussian(double sigma, double x0, double y0,
                   const int nchan, const int nstokes, const int nrow,
                   double* u, double* v,
                   double* size,
				   double* pos_real, double* pos_imag,
                   const DataContainer data);
void calc_disk(double sigma, double x0, double y0,
               const int nchan, const int nstokes, const int nrow,
               double* u, double* v,
               double* size, double* dsize_dsigma,
			   double* pos_real, double* pos_imag,
               const DataContainer data);


#endif // __COMMON_CUDA_H__
