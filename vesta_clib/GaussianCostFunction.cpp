#include "GaussianCostFunction.h"

bool GaussianCostFunction::Evaluate(double const* const* parameters,
                                    double* residuals,
                                    double** jacobians) const
{
	double flux         = parameters[0][0];
	double x0           = parameters[1][0];
	double y0           = parameters[2][0];
	double sigma_x      = parameters[3][0];
	double sigma_y      = parameters[4][0];
	double PA           = parameters[5][0];

	double cosPA        = cos(PA);
	double sinPA        = sin(PA);
	double omega_x      = (_u*cosPA-_v*sinPA)*sigma_x;
	double omega_y      = (_u*sinPA+_v*cosPA)*sigma_y;
	double size         = exp(-2*M_PI*M_PI*(omega_x*omega_x+omega_y*omega_y));
	double pos_real     = cos(-2*M_PI*(x0*_u+y0*_v));
	double pos_imag     = sin(-2*M_PI*(x0*_u+y0*_v));
	double V_mod_real   = flux*size*pos_real;
	double V_mod_imag   = flux*size*pos_imag;

	residuals[0] = _V_real - V_mod_real;
	residuals[1] = _V_imag - V_mod_imag;

	if(jacobians != NULL)
	{
		if(jacobians[0] != NULL)
		{
			jacobians[0][0] = -size*pos_real;
			jacobians[0][1] = -size*pos_imag;
		}
		if(jacobians[1] != NULL)
		{
			jacobians[1][0] = -2*M_PI*_u*V_mod_imag;
			jacobians[1][1] = 2*M_PI*_u*V_mod_real;
		}
		if(jacobians[2] != NULL)
		{
			jacobians[2][0] = -2*M_PI*_v*V_mod_imag;
			jacobians[2][1] = 2*M_PI*_v*V_mod_real;
		}
		if(jacobians[3] != NULL)
		{
			jacobians[3][0] = V_mod_real * 2*M_PI*M_PI * 2*omega_x * (_u*cosPA-_v*sinPA);
			jacobians[3][1] = V_mod_imag * 2*M_PI*M_PI * 2*omega_x * (_u*cosPA-_v*sinPA);
		}
		if(jacobians[4] != NULL)
		{
			jacobians[4][0] = V_mod_real * 2*M_PI*M_PI * 2*omega_y * (_u*sinPA+_v*cosPA);
			jacobians[4][1] = V_mod_imag * 2*M_PI*M_PI * 2*omega_y * (_u*sinPA+_v*cosPA);
		}
		if(jacobians[5] != NULL)
		{
			double domega_x_dPA = (-_u*sinPA-_v*cosPA)*sigma_x;
			double domega_y_dPA = ( _u*cosPA-_v*sinPA)*sigma_y;
			jacobians[5][0] = V_mod_real * 2*M_PI*M_PI * (2*omega_x*domega_x_dPA + 2*omega_y*domega_y_dPA);
			jacobians[5][1] = V_mod_imag * 2*M_PI*M_PI * (2*omega_x*domega_x_dPA + 2*omega_y*domega_y_dPA);
		}
	}
	return true;
}
