#pragma once

#include <cuda_runtime.h>

__device__ uint32_t dilate10(const uint32_t value) {
	uint32_t x;
	x = value & 0x03FF;
	x = ((x << 16) + x) & 0xFF0000FF;
	x = ((x << 8) + x) & 0x0F00F00F;
	x = ((x << 4) + x) & 0xC30C30C3;
	x = ((x << 2) + x) & 0x49249249;
	return x;
}

__device__ uint64_t dilate20(const uint32_t value) {
	uint32_t lo = value & 0x3FF;
	uint32_t hi = (value >> 10) & 0x3FF;

	uint64_t dlo = dilate10(lo);
	uint64_t dhi = dilate10(hi);

	return dlo | (dhi << 30);
}

__global__ void computeMortonKeys(const float4* __restrict__ values, const int len, const float domainMin, const float domainMax, uint64_t* __restrict__ keys)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= len) return;

	float3 v = make_float3(values[idx].x, values[idx].y, values[idx].z);
	v.x -= domainMin;
	v.y -= domainMin;
	v.z -= domainMin;

	float scale = (float)(1 << 20) / (domainMax - domainMin);

	uint64_t dx = dilate20((uint32_t)(v.x * scale));
	uint64_t dy = dilate20((uint32_t)(v.y * scale));
	uint64_t dz = dilate20((uint32_t)(v.z * scale));

	uint64_t key = dx | (dy << 1) | (dz << 2);

	keys[idx] = key;
}

__global__ void getMaskedValues(const uint64_t* __restrict__ keys, const int len, const int level, uint64_t* __restrict__ masked_keys)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= len) return;

	int bits = 3 * (level + 1);
	uint64_t bitMask = (1 << bits) - 1;
	int shift = 60 - bits;
	masked_keys[idx] = keys[idx] & (bitMask << shift);
}