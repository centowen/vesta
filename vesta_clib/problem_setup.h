#include <iostream>
#include <complex>
#include <cmath>
#include <ceres/ceres.h>
#include <glog/logging.h>

#include "Chunk.h"


using ceres::Problem;
using std::string;


const int chunk_size = 1000000;


void add_residual_blocks(Problem& problem, string path);
void add_chunk_to_residual_blocks(Problem& problem, Chunk& chunk);
