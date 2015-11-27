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
#include <ceres/ceres.h>
#include <cmath>
using ceres::CostFunction;

#ifdef ENABLE_CUDA
#include "CommonCuda.h"
#endif


#ifndef __DISK_COST_H__
#define __DISK_COST_H__

class DiskCost : public CostFunction {/*{{{*/
private:
	double* _u;
	double* _v;
	double* size;
	double* dsize_dsigma;
	double* pos_real;
	double* pos_imag;

#ifdef ENABLE_CUDA
	DataContainer cu_data;
#endif

	double* _V_real;
	double* _V_imag;
	double* sqrt_weights;

	bool* _flags;

	int _nchan;
	int _nstokes;
	int _nrow;

public:
	DiskCost(float* u, float* v, float* V_real, float* V_imag,
	                     float* weights, int* flags,
	                     const int nchan, const int nstokes, const int nrow);
	virtual ~DiskCost();

	virtual bool Evaluate(double const* const* parameters,
	                      double* residuals,
	                      double** jacobians) const;
};/*}}}*/
#endif
