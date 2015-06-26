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

#include "problem_setup.h"

int main(int argc, char* argv[])
{
	double flux;
	double sigma;
	double x0;
	double y0;
	double flux_point_source;

	ceres::Problem problem;

	add_residual_blocks(problem, std::string("/data2/lindroos/ecdfs/aless/stack/k20faint1/stack.uv.ms"),
	                    flux, sigma, x0, y0, flux_point_source);
// 	add_residual_blocks(problem, std::string("/data2/lindroos/ecdfs/aless/stack/k20/stack.uv.ms"),
// 	                    flux, sigma, x0, y0, flux_point_source);
// 	add_residual_blocks(problem, std::string("/home/lindroos/data/k20.stack.uv.ms"),
// 	                    flux, sigma, x0, y0);

	ceres::Solver::Options options;
	options.max_num_iterations = 50;

	// Achieves the best performance in a simple test
	// ~2 times faster than DENSE_QR
	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;

// 	options.linear_solver_type = ceres::DENSE_QR;
// 	options.linear_solver_type = ceres::DENSE_SCHUR;

// 	Sparse solvers is a very bad option for this problem.
//  Unless possibly fitting x0, y0 and changing residuals
//  to be calculated in abs, phase in place of real, imag
// 	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = false;
	options.function_tolerance = 1e-14;
	options.num_threads = 32;
	options.num_linear_solver_threads = 32;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

// 	double cost;
// 	problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
// 	std::cout << "flux = " << flux*1e3 << "mJy" << std::endl;
	std::cout << "sigma = " << sigma/M_PI*180.*3600. << "arcsec, flux = "
	          << flux*1e3 << "mJy";
// 	std::cout << " , flux (ps): " << flux_point_source*1e3 << "mJy";
	std::cout << std::endl;
	std::cout << "x0 = " << x0 << ", y0 = " << y0 << std::endl;
}
