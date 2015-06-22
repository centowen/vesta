#include <iostream>
#include <fstream>
#include <complex>
#include <cmath>
#include <ceres/ceres.h>
#include <glog/logging.h>


using namespace std;
using ceres::AutoDiffCostFunction;
using ceres::DynamicAutoDiffCostFunction;
using ceres::CostFunction;
using ceres::CauchyLoss;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

double fftfreq(int n, int N, double d = 1.0)
{
	double f = 0.;
	if(n< N/2)
		f = double(n)/double(N)*d;
	else
		f = (double(n)-double(N))/double(N)*d;
	return f;
}

class GaussResidualMulti : public CostFunction {/*{{{*/
private:
	double _u, _v, _V_real, _V_imag;
	int _n_gauss;
	double _beam_sigma_x, _beam_sigma_y, _beam_PA;

public:
	GaussResidualMulti(double u, double v, double V_real, double V_imag, int n_gauss,
			           double beam_sigma_x, double beam_sigma_y, double beam_PA)
		:_u(u), _v(v), _V_real(V_real), _V_imag(V_imag), _n_gauss(n_gauss),
		_beam_sigma_x(beam_sigma_x), _beam_sigma_y(beam_sigma_y),
		_beam_PA(beam_PA)
	{
		for (int i = 0; i < 6; ++i) {
			mutable_parameter_block_sizes()->push_back(_n_gauss);
		}   
		set_num_residuals(2);
	}
	virtual ~GaussResidualMulti() {}

	virtual bool Evaluate(double const* const* parameters,
	                      double* residuals,
						  double** jacobians) const{
		double flux;
		double x0;
		double y0;
		double sigma_x;
		double sigma_y;
		double PA;

		double cosPA;
		double sinPA;
		double beam_omega_x;
		double beam_omega_y;
		double beam;
		double omega_x;
		double omega_y;
		double size;
		double pos_real;
		double pos_imag;
		double V_mod_real;
		double V_mod_imag;

		residuals[0] = _V_real;
		residuals[1] = _V_imag;

		for(int i = 0; i < _n_gauss; i++){
			flux         = parameters[0][i];
			x0           = parameters[1][i];
			y0           = parameters[2][i];
			sigma_x      = parameters[3][i];
			sigma_y      = parameters[4][i];
			PA           = parameters[5][i];

			cosPA        = cos(PA);
			sinPA        = sin(PA);
			beam_omega_x = (_u*cos(_beam_PA)-_v*sin(_beam_PA))*_beam_sigma_x;
			beam_omega_y = (_u*sin(_beam_PA)+_v*cos(_beam_PA))*_beam_sigma_y;
// 			beam         = 1.;
			beam         = exp(-2*M_PI*M_PI*(beam_omega_x*beam_omega_x+
			                                 beam_omega_y*beam_omega_y))*
			                                 (2.*M_PI*_beam_sigma_x*_beam_sigma_y);
			omega_x      = (_u*cosPA-_v*sinPA)*sigma_x;
			omega_y      = (_u*sinPA+_v*cosPA)*sigma_y;
			size         = exp(-2*M_PI*M_PI*(omega_x*omega_x+omega_y*omega_y));
			pos_real     = cos(-2*M_PI*(x0*_u+y0*_v));
			pos_imag     = sin(-2*M_PI*(x0*_u+y0*_v));
			//V_mod_real   = flux*beam*size*pos_real;
			//V_mod_imag   = flux*beam*size*pos_imag;
			V_mod_real   = flux*beam*size*pos_real;
			V_mod_imag   = flux*beam*size*pos_imag;

			residuals[0] -= V_mod_real;
			residuals[1] -= V_mod_imag;

			if(jacobians != NULL)
			{
				if(jacobians[0] != NULL)
				{
					jacobians[0][i+_n_gauss*0] = -beam*size*pos_real;
					jacobians[0][i+_n_gauss*1] = -beam*size*pos_imag;
				}
				if(jacobians[1] != NULL)
				{
					jacobians[1][i+_n_gauss*0] = -2*M_PI*_u*V_mod_imag;
					jacobians[1][i+_n_gauss*1] = 2*M_PI*_u*V_mod_real;
				}
				if(jacobians[2] != NULL)
				{
					jacobians[2][i+_n_gauss*0] = -2*M_PI*_v*V_mod_imag;
					jacobians[2][i+_n_gauss*1] = 2*M_PI*_v*V_mod_real;
				}
				if(jacobians[3] != NULL)
				{
					jacobians[3][i+_n_gauss*0] = V_mod_real * 2*M_PI*M_PI * 2*omega_x * (_u*cosPA-_v*sinPA);
					jacobians[3][i+_n_gauss*1] = V_mod_imag * 2*M_PI*M_PI * 2*omega_x * (_u*cosPA-_v*sinPA);
				}
				if(jacobians[4] != NULL)
				{
					jacobians[4][i+_n_gauss*0] = V_mod_real * 2*M_PI*M_PI * 2*omega_y * (_u*sinPA+_v*cosPA);
					jacobians[4][i+_n_gauss*1] = V_mod_imag * 2*M_PI*M_PI * 2*omega_y * (_u*sinPA+_v*cosPA);
				}
				if(jacobians[5] != NULL)
				{
					double domega_x_dPA = (-_u*sinPA-_v*cosPA)*sigma_x;
					double domega_y_dPA = ( _u*cosPA-_v*sinPA)*sigma_y;
					jacobians[5][i+_n_gauss*0] = V_mod_real * 2*M_PI*M_PI * (2*omega_x*domega_x_dPA + 2*omega_y*domega_y_dPA);
					jacobians[5][i+_n_gauss*1] = V_mod_imag * 2*M_PI*M_PI * (2*omega_x*domega_x_dPA + 2*omega_y*domega_y_dPA);
				}
			}
		}
		return true;
	}

};/*}}}*/


void fit_cpp(double* data, int N, double* fitpar, int n_gauss, /*{{{*/
		     double beam_sigma_x, double beam_sigma_y, double beam_PA)
{
// 	google::InitGoogleLogging("peti");
	Problem problem;

	double* flux    = new double[n_gauss];
	double* x0      = new double[n_gauss];
	double* y0      = new double[n_gauss];
	double* sigma_x = new double[n_gauss];
	double* sigma_y = new double[n_gauss];
	double* PA      = new double[n_gauss];

	for(int i = 0; i < n_gauss; i++)
	{
		flux   [i] = fitpar[0+6*i];
		x0     [i] = fitpar[1+6*i];
		y0     [i] = fitpar[2+6*i];
		sigma_x[i] = fitpar[3+6*i];
		sigma_y[i] = fitpar[4+6*i];
		PA     [i] = fitpar[5+6*i];
	}

	double cost = 0.;
	double* res = new double[2];
	double** pars = new double*[6];
	for(int u = 0; u < N; u++)
		for(int v = 0; v < N; v++)
		{
			CostFunction* cost_function = new GaussResidualMulti(fftfreq(u,N), fftfreq(v,N),
					                                             data[0+u*2+v*2*N], data[1+u*2+v*2*N], 
																 n_gauss, 
																 beam_sigma_x, beam_sigma_y, beam_PA);
			pars[0] = flux;
			pars[1] = x0;
			pars[2] = y0;
			pars[3] = sigma_x;
			pars[4] = sigma_y;
			pars[5] = PA;

			cost_function->Evaluate(pars, res, NULL);
			cost += 0.5*res[0]*res[0];
			cost += 0.5*res[1]*res[1];

			problem.AddResidualBlock(cost_function, NULL,
			                         flux, x0, y0, 
									 sigma_x, sigma_y, PA);
		}

	for(int i = 0; i < n_gauss; i++)
	{
		problem.SetParameterLowerBound(flux,    i, 0.);
		problem.SetParameterLowerBound(x0,      i, 2.);
		problem.SetParameterLowerBound(y0,      i, 2.);
		problem.SetParameterLowerBound(sigma_x, i, 0.);
		problem.SetParameterLowerBound(sigma_y, i, 0.);
		problem.SetParameterLowerBound(PA, i, double(0));
		problem.SetParameterUpperBound(x0,      i, double(N-1));
		problem.SetParameterUpperBound(y0,      i, double(N-1));
		problem.SetParameterUpperBound(sigma_x, i, double(N)/2);
		problem.SetParameterUpperBound(sigma_y, i, double(N)/2);
		problem.SetParameterUpperBound(PA, i, double(M_PI));
	}


	Solver::Options options;
	options.max_num_iterations = 5000;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = false;

	Solver::Summary summary;
	Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";
	std::cout << "cost: " << cost << endl;

	for(int i = 0; i < n_gauss; i++)
	{
		fitpar[0+6*i] = flux   [i];
		fitpar[1+6*i] = x0     [i];
		fitpar[2+6*i] = y0     [i];
		fitpar[3+6*i] = sigma_x[i];
		fitpar[4+6*i] = sigma_y[i];
		fitpar[5+6*i] = PA     [i];
	}
}/*}}}*/

extern "C"{
double* fit(double* data, int N, double* guess, int n_gauss,
		double beam_sigma_x, double beam_sigma_y, double beam_PA)
{
	double* fitpar = new double[6*n_gauss];

	for(int i = 0; i < n_gauss*6; i++)
	{
		fitpar[i] = guess[i];
	}
	
	fit_cpp(data, N, fitpar, n_gauss, 
			beam_sigma_x, beam_sigma_y, beam_PA);
	return fitpar;
}
}

