#include "NBodySim.h"

#include <cuda_runtime.h>
#include <cuda/cmath>
#include <iostream>

#include <random>

#include "NBodySimKernels.h"
#include "utils.h"

const int domainMin = 0;
const int domainMax = 1000;
const int Nleaf = 16;

NBodySim::NBodySim(int bodyCount)
{
	_bodyCount = bodyCount;

	cudaMallocHost(&_h_particleInfos, _bodyCount * sizeof(float4));
	cudaMalloc(&_d_particleInfos, _bodyCount * sizeof(float4));
	cudaMalloc(&_keys, _bodyCount * sizeof(uint64_t));
	cudaMalloc(&_maskedKeys, _bodyCount * sizeof(uint64_t));

	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<float> posDist(domainMin, domainMax);
	std::uniform_real_distribution<float> massDist(100, 200);
	for (int i = 0; i < bodyCount; i++)
	{
		_h_particleInfos[i] = make_float4(posDist(rng), posDist(rng), posDist(rng), massDist(rng));
	}

	cudaMemcpy(_d_particleInfos, _h_particleInfos, bodyCount * sizeof(float4), cudaMemcpyDefault);
}

void NBodySim::Simulate()
{
	int threadsPerBlock = 256;
	int blocks = cuda::ceil_div(_bodyCount, threadsPerBlock);
	computeMortonKeys<<<blocks, threadsPerBlock>>>(_d_particleInfos, _bodyCount, domainMin, domainMax, _keys);

	cudaDeviceSynchronize();

	void* kernelArgs[] = { &_d_particleInfos, &_keys, &_bodyCount };
	cudaLaunchCooperativeKernel(radixSortByKey<float4, uint64_t>, dim3(blocks), dim3(threadsPerBlock), kernelArgs);

	cudaDeviceSynchronize();

	bool* flagged;
	std::vector<int> activeList;
	activeList.reserve(_bodyCount);
	for (int i = 0; i < _bodyCount; i++)
	{
		activeList.push_back(i);
	}
	int level = 0;
	while (!activeList.empty() && level < 20)
	{
		getMaskedValues<<<blocks, threadsPerBlock>>>(_keys, _bodyCount, level, _maskedKeys);
		cudaDeviceSynchronize();
		level++;
	}
}

void NBodySim::Render(uchar4* pbo)
{
}
