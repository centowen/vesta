#include "msio.h"

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
		float* freq = chunk.inVis->freq;
		for(int chan = 0; chan < chunk.nChan(); chan++)
		{
			if(chan==2 and uvrow ==7)
			{
				cout << "Val = " << freq[chan]/1e9 << endl;
			}
		}
	}
}
