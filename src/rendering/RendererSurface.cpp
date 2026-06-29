#include "RenderSurface.h"

#include <cuda_gl_interop.h>

#include "shader_utils.h"

RenderSurface::RenderSurface(int width, int height)
{
	_width = width;
	_height = height;
	_glProgram = 0;
	_pbo = 0;
	_tex = 0;
	_vao = 0;
	_cudaPBO = nullptr;

	InitGLResources();
}

RenderSurface::~RenderSurface()
{
	Cleanup();
}

void RenderSurface::Draw()
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);

	glBindTexture(GL_TEXTURE_2D, _tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glUseProgram(_glProgram);
	glBindVertexArray(_vao);

	glActiveTexture(GL_TEXTURE0);

	GLint loc = glGetUniformLocation(_glProgram, "screenTexture");
	if (loc != -1) {
		glUniform1i(loc, 0);
	}

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	glBindVertexArray(0);
	glUseProgram(0);
}

void RenderSurface::Cleanup()
{
	if (_cudaPBO) {
		cudaGraphicsUnregisterResource(_cudaPBO);
		_cudaPBO = nullptr;
	}
	if (_pbo) {
		glDeleteBuffers(1, &_pbo);
		_pbo = 0;
	}
	if (_tex) {
		glDeleteTextures(1, &_tex);
		_tex = 0;
	}
	if (_glProgram) {
		glDeleteProgram(_glProgram);
		_glProgram = 0;
	}
	if (_vao) {
		glDeleteVertexArrays(1, &_vao);
		_vao = 0;
	}
}

uchar4* RenderSurface::MapCudaResource()
{
	cudaGraphicsMapResources(1, &_cudaPBO, 0);

	uchar4* dptr;
	size_t size;
	cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, _cudaPBO);

	return dptr;
}

void RenderSurface::UnmapCudaResource()
{
	cudaGraphicsUnmapResources(1, &_cudaPBO, 0);
}

void RenderSurface::InitGLResources()
{
	glGenBuffers(1, &_pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, _width * _height * 4 * sizeof(uint8_t), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(&_cudaPBO, _pbo, cudaGraphicsMapFlagsWriteDiscard);

	glGenTextures(1, &_tex);
	glBindTexture(GL_TEXTURE_2D, _tex);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, _width, _height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	_glProgram = createProgram("src/shaders/quad.vert", "src/shaders/quad.frag");

	glGenVertexArrays(1, &_vao);
	glBindVertexArray(_vao);
	glBindVertexArray(0);
}
