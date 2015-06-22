#include <ceres/ceres.h>

#include "problem_setup.h"

int main(int argc, char* argv[])
{
	ceres::Problem problem;
	add_residual_blocks(problem, std::string("/home/lindroos/data/k20.stack.uv.ms"));
}
