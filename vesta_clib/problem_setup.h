// vesta, use Ceres to do uv-model fitting for ms-data.
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

const int mod_gaussian = 0;
const int mod_gaussian_ps = 1;
const int mod_ps = 2;
const int mod_disk = 3;
const int mod_disk_ps = 4;


const int chunk_size = 1000000;
const float C_LIGHT = 299792458;


void add_residual_blocks(Problem& problem, string path,
                         double& flux, double& sigma, double& x0, double& y0,
                         double& flux_point_source, int model);
void add_chunk_to_residual_blocks(Problem& problem, Chunk& chunk,
                                  double& flux, double& sigma,
                                  double& x0, double& y0,
                                  double& flux_point_source,
								  int model);
// void add_chunk_to_residual_blocks(Problem& problem, Chunk& chunk);
#endif
