#pragma once

#include <cuda_runtime.h>

class NBodySim
{
public:
	NBodySim(int bodyCount);
	void Simulate();
	void Render(uchar4* pbo);

private:
	int _bodyCount = 0;
	float4* _h_particleInfos = nullptr;
	float4* _d_particleInfos = nullptr;
	long* _keys = nullptr;
	long* _maskedKeys = nullptr;
};