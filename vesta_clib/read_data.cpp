#include <iostream>
#include "DataIO.h"
#include "msio.h"
#include "Chunk.h"

using std::vector;
using std::string;

void read_data(vector<Chunk>& chunks, string path)
{
	DataIO data(path.c_str(), "", msio::col_corrected_data);
	while(data->readChunk(chunk))
	{
		chunks.push_bask(chunk);
	}
}
