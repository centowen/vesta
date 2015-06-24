#include <ceres/ceres.h>

#include "problem_setup.h"

int main(int argc, char* argv[])
{
	double flux;
	double sigma;
	double x0;
	double y0;

	ceres::Problem problem;
	add_residual_blocks(problem, std::string("/home/lindroos/data/k20.stack.uv.ms"),
	                    flux, sigma, x0, y0);

	ceres::Solver::Options options;
	options.max_num_iterations = 10;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = false;
	options.function_tolerance = 1e-12;
	options.num_threads = 4;
	options.num_linear_solver_threads = 4;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

// 	double cost;
// 	problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
	std::cout << "sigma = " << sigma/M_PI*180.*3600. << "arcsec, flux = " 
	          << flux*1e3 << "mJy" << std::endl;
// 
// 	sigma = 0.;
// 	problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
// 	std::cout << "sigma = " << sigma << ": cost = " << cost << std::endl;
// 
// 	sigma = 4.848e-6;
// 	flux = 1e-2;
// 	problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
// 	std::cout << "flux = " << flux << ": cost = " << cost << std::endl;
// 
// 	sigma = 4.848e-6;
// 	flux = 1e-3;
// 	problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
// 	std::cout << "flux = " << flux << ": cost = " << cost << std::endl;
// 
// 	sigma = 4.848e-6;
// 	flux = 0.;
// 	problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);
// 	std::cout << "flux = " << flux << ": cost = " << cost << std::endl;
}
