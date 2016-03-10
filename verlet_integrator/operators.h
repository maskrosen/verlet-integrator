#ifndef OPERATORS_H
#define OPERATORS_H
#endif

#include "cuda_runtime.h"

__device__ float3 operator+(const float3 &a, const float3 &b) {

	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}

__device__ float3 operator-(const float3 &a, const float3 &b) {

	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);

}

__device__ float3 operator+(const float3 &a, const float &b) {

	return make_float3(a.x + b, a.y + b, a.z + b);

}

__device__ float3 operator*(const float3 &a, const float &b) {

	return make_float3(a.x * b, a.y * b, a.z * b);

}