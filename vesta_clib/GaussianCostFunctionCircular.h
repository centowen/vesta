#include <ceres/ceres.h>
#include <cmath>
using ceres::CostFunction;



#ifndef __GAUSSIAN_COST_FUNCTION_CIRCULAR_H__
#define __GAUSSIAN_COST_FUNCTION_CIRCULAR_H__

class GaussianCostFunctionCircular : public CostFunction {/*{{{*/
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
	GaussianCostFunctionCircular(float* u, float* v, float* V_real, float* V_imag,
	                     float* weights, bool* flags,
	                     int nchan, int nstokes);
	virtual ~GaussianCostFunctionCircular();

	virtual bool Evaluate(double const* const* parameters,
	                      double* residuals,
	                      double** jacobians) const;
};/*}}}*/
#endif
