#include "msio.h"
#include "problem_setup.h"
#include "GaussianCostFunction.h"

// using ceres::Problem;
// using std::string;
using std::cout;
using std::endl;

void add_residual_blocks(Problem& problem, string path)
{
	Chunk chunk(chunk_size);
	DataIO* data = new msio(path.c_str(), "", msio::col_corrected_data);
	while(data->readChunk(chunk))
	{
		add_chunk_to_residual_blocks(problem, chunk);
	}
	delete data;
}

void add_chunk_to_residual_blocks(Problem& problem, Chunk& chunk)
{
	for(int uvrow = 0; uvrow < chunk.size(); uvrow++)
	{
		Visibility& inVis = chunk.inVis[uvrow];
		float* freq = inVis.freq;


		for(int chan = 0; chan < chunk.nChan(); chan++)
		{
			for(int i_stokes = 0; i_stokes < chunk.nStokes(); i_stokes++)
			{
				size_t data_index = i_stokes*chunk.nChan()+chan;
				float V_real = inVis.data_real[data_index ];
				float V_imag = inVis.data_imag[data_index ];
				float weight = inVis.weight[i_stokes];
				CostFunction* cost_function = 
					new GaussianCostFunction(inVis.u * freq[chan] / C_LIGHT,
					                         inVis.v * freq[chan] / C_LIGHT,
					                         V_real, V_imag);
			}
		}
	}
}
