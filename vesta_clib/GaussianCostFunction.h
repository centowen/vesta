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



#ifndef __GAUSSIAN_COST_FUNCTION_H__
#define __GAUSSIAN_COST_FUNCTION_H__


class GaussianCostFunction : public CostFunction {/*{{{*/
private:
	double* _u;
	double* _v;

	double* _V_real;
	double* _V_imag;
	double* sqrt_weights;

	bool* _flags;

	int _nchan;
	int _nstokes;

public:
	GaussianCostFunction(float* u, float* v, float* V_real, float* V_imag,
	                     float* weights, bool* flags,
	                     int nchan, int nstokes);
	virtual ~GaussianCostFunction();

	virtual bool Evaluate(double const* const* parameters,
	                      double* residuals,
	                      double** jacobians) const;
};/*}}}*/
#endif
