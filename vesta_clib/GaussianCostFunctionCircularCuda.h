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

#ifndef __GAUSSIAN_COST_FUNCTION_CIRCULAR_CUDA_H__
#define __GAUSSIAN_COST_FUNCTION_CIRCULAR_CUDA_H__

void allocate_stuff(const int nchan, const int nstokes);
void free_stuff();
void calc_functions(double sigma, double x0, double y0, const int nchan, const int nstokes,
                    double* u, double* v,
					double* size, double* pos_real, double* pos_imag);
#endif
