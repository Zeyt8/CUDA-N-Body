#pragma once

#include <cuda_runtime.h>

#include <GL/glew.h>

class RenderSurface {
public:
	RenderSurface(int width, int height);
	~RenderSurface();

	void Draw();
	void Cleanup();
	uchar4* MapCudaResource();
	void UnmapCudaResource();

private:
	void InitGLResources();

private:
	int _width, _height;
	GLuint _glProgram;
	GLuint _pbo;
	GLuint _tex;
	cudaGraphicsResource* _cudaPBO;
	GLuint _vao;
};
