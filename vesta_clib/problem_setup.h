#include <iostream>
#include <complex>
#include <cmath>
#include <ceres/ceres.h>
#include <glog/logging.h>

#include "Chunk.h"
#ifndef __PROBLEM_SETUP_H__
#define __PROBLEM_SETUP_H__
using ceres::Problem;
using std::string;


const int chunk_size = 1000000;
const float C_LIGHT = 299792458;


void add_residual_blocks(Problem& problem, string path);
void add_chunk_to_residual_blocks(Problem& problem, Chunk& chunk);
#endif
