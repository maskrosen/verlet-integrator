
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "operators.h"


cudaError_t calcVerletStep(float3 *p, float3 *pOld, float3 *pTemp, float3 a, float dt, unsigned int size);
void print(float3 v);
void initializePositions(float3 *p, float3 *pOld, int size);

__global__ void verletKernel(float3 *p, float3 *pOld, float3 *pTemp, float3 a, float dt)
{
    int i = threadIdx.x;
	pTemp[i] = p[i];
    p[i] = p[i] + p[i] - pOld[i] + a* dt*dt;
	pOld[i] = pTemp[i];
}

int main()
{
    const int arraySize = 10000;
	float3 p[arraySize] = { make_float3(1, 1, 1), make_float3(2, 1, 1), make_float3(3, 1, 1), make_float3(4, 1, 1), make_float3(5, 1, 1) };
	float3 pOld[arraySize] = { make_float3(1, 1, 1), make_float3(2, 1, 1), make_float3(3, 1, 1), make_float3(4, 1, 1), make_float3(5, 1, 1) };
	float3 pTemp[arraySize] = { 0 };
	float3 a = make_float3(0, -9.82f, 0);
  
	initializePositions(p, pOld, arraySize);


    // Add vectors in parallel.
	cudaError_t cudaStatus = calcVerletStep(p, pOld, pTemp, a, 0.016f ,arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	
	/*
	printf("Result: ");

	for (const float3 &position : p){
		print(position);
	}

	for (const float3 &position : pOld){
		print(position);
	}
	*/

	print(p[arraySize-1]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

void initializePositions(float3 *p, float3 *pOld, int size){
	for (int i = 0; i < size; i++){
		p[i] = make_float3(i, 1, 1);
		pOld[i] = make_float3(i, 1, 1);
	}
}

void update(float timestep) {
	//Update the positions and velocities
	// xi+1 = xi + (xi - xi-1) + a * dt * dt


}

void print(float3 v){
	printf("(%f, %f, %f)", v.x, v.y, v.z);
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t calcVerletStep(float3 *p, float3 *pOld, float3 *pTemp, float3 a, float dt, unsigned int size)
{
    float3 *dev_p = 0;
	float3 *dev_pOld = 0;
	float3 *dev_pTemp = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_p, size * sizeof(float3));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_pOld, size * sizeof(float3));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	
    cudaStatus = cudaMalloc((void**)&dev_pTemp, size * sizeof(float3));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_p, p, size * sizeof(float3), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_pOld, pOld, size * sizeof(float3), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	verletKernel <<<size/16, 16 >>>(dev_p, dev_pOld, dev_pTemp, a, dt);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(p, dev_p, size * sizeof(float3), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(pOld, dev_pOld, size * sizeof(float3), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
    cudaFree(dev_p);
    cudaFree(dev_pOld);
    cudaFree(dev_pTemp);
    
    return cudaStatus;
}
