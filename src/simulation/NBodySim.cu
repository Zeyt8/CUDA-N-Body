#include "NBodySim.h"

#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include <cuda/cmath>
#include <iostream>
#include <cmath>

#include <random>

const int domainMin = 0;
const int domainMax = 1000;
const int Nleaf = 16;

NBodySim::NBodySim(int bodyCount)
{
	_bodyCount = bodyCount;

	cudaMallocHost(&_h_particleInfos, _bodyCount * sizeof(float4));
	cudaMalloc(&_d_particleInfos, _bodyCount * sizeof(float4));
	cudaMalloc(&_keys, _bodyCount * sizeof(long));
	cudaMalloc(&_maskedKeys, _bodyCount * sizeof(long));

	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<float> posDist(domainMin, domainMax);
	std::uniform_real_distribution<float> massDist(100, 200);
	for (int i = 0; i < bodyCount; i++)
	{
		_h_particleInfos[i] = float4(posDist(rng), posDist(rng), massDist(rng), 0);
	}

	cudaMemcpy(_d_particleInfos, _h_particleInfos, bodyCount * sizeof(float4), cudaMemcpyDefault);
}

__device__ int dilate(const int value) {
	unsigned int x;
	x = value & 0x03FF;
	x = ((x << 16) + x) & 0xFF0000FF;
	x = ((x << 8) + x) & 0x0F00F00F;
	x = ((x << 4) + x) & 0xC30C30C3;
	x = ((x << 2) + x) & 0x49249249;
	return x;
}

__global__ void computeMortonKeys(const float4* __restrict__ values, const int len, const int domainMin, const int domainMax, long* __restrict__ keys)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= len) return;

	float2 v = float2(values[idx].x, values[idx].y);
	v.x = (v.x - domainMin) * (domainMax - domainMin);
	v.y = (v.y - domainMin) * (domainMax - domainMin);

	long key = dilate(v.x) | (dilate(v.y) << 1);

	keys[idx] = key;
}

template<typename T, typename Key>
__device__ void partitionByBit(T* __restrict__ values, Key* __restrict__ keys, const int len, const int bit)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	bool active = idx < len;
	cg::grid_group g = cg::this_grid();

	T valueBefore;
	Key keyBefore;
	int b_i;
	if (active)
	{
		valueBefore = values[idx];
		keyBefore = keys[idx];
		b_i = (keyBefore >> bit) & 1;
		keys[idx] = b_i;
	}

	g.sync();

	for (int i = 1; i < len; i *= 2)
	{
		Key b = 0;
		if (idx >= i)
		{
			b = keys[idx - i];
		}
		g.sync();
		if (idx >= i)
		{
			keys[idx] += b;
		}
		g.sync();
	}

	if (active)
	{
		int zeroTotal = len - keys[len - 1];
		int oneBefore = keys[idx];

		if (b_i)
		{
			values[zeroTotal + oneBefore - 1] = valueBefore;
		}
		else
		{
			values[idx - oneBefore] = valueBefore;
		}
		keys[idx] = keyBefore;
	}
}

template<typename T, typename Key>
__global__ void radixSortByKey(T* __restrict__ values, Key* __restrict__ keys, const int len)
{
	cg::grid_group g = cg::this_grid();
	for (int bit = 0; bit < 32; bit++)
	{
		partitionByBit(values, keys, len, bit);
		g.sync();
	}
}

__global__ void partitionByPosition(const long* __restrict__ keys, const int len, const int level, long* __restrict__ masked_keys)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= len) return;

	int bits = 3 * (level + 1);
	long bitMask = (1 << bits) - 1;
	int shift = 60 - bits;
	masked_keys[idx] = keys[idx] & (bitMask << shift);
}

void NBodySim::Simulate()
{
	int threadsPerBlock = 256;
	int blocks = cuda::ceil_div(_bodyCount, threadsPerBlock);
	computeMortonKeys<<<blocks, threadsPerBlock>>>(_d_particleInfos, _bodyCount, domainMin, domainMax, _keys);

	cudaDeviceSynchronize();

	void* kernelArgs[] = { &_d_particleInfos, &_keys, &_bodyCount };
	radixSortByKey<float4, long><<<blocks, threadsPerBlock>>>(_d_particleInfos, _keys, _bodyCount);
	//cudaLaunchCooperativeKernel(radixSortByKey<float4, long>, dim3(blocks), dim3(threadsPerBlock), kernelArgs);

	cudaDeviceSynchronize();

	bool* flagged;
	std::vector<int> activeList;
	int level = 0;
	while (!activeList.empty() && level < 20)
	{
		partitionByPosition<<<blocks, threadsPerBlock>>>(_keys, _bodyCount, level, _maskedKeys);
		cudaDeviceSynchronize();
		level++;
	}
}

void NBodySim::Render(uchar4* pbo)
{
}
