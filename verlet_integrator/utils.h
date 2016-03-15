#pragma once

#include <stdio.h>
#include "cuda_runtime.h"
#include <gl/glew.h>
#include <GL/freeglut.h> 


void print(float3 v);

int iDivUp(int a, int b);

const char *textFileRead(const char *fn);

GLuint loadShaders();
