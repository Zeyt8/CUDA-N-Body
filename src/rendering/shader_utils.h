#pragma once

#include <string>
#include <GL/glew.h>

std::string loadShaderSource(const std::string& path);

GLuint compileShader(GLenum type, const std::string& source);

GLuint createProgram(const std::string& vertexPath, const std::string& fragmentPath);