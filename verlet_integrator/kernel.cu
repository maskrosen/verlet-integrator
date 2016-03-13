
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "operators.h"
#include <gl/glew.h>
#include <GL/freeglut.h> 
#include <math.h>
#include <ctime>


cudaError_t calcVerletStep(float3 *p, float3 *pOld, float3 *pTemp, float3 a, float dt, unsigned int size);
void print(float3 v);
void initializePositions(float3 *p, float3 *pOld, int size);

const unsigned int window_width = 1024;
const unsigned int window_height = 1024;

const int arraySize = 5000;
float3 *p = new float3[arraySize];
float3 *pOld = new float3[arraySize];
float3 *pTemp = new float3[arraySize];

float3 a = make_float3(0, -2.0f, 0);
int currentTime = 0;
int previousTime = 0;

__global__ void verletKernel(float3 *p, float3 *pOld, float3 *pTemp, float3 a, float dt)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	pTemp[i] = p[i];
    p[i] = p[i] + p[i] - pOld[i] + a* dt*dt;
	pOld[i] = pTemp[i];

	//Collision against floor
	if (p[i].y < 0){

		//Bouncing collision
		pTemp[i].y = p[i].y;
		p[i].y = pOld[i].y*0.5;
		pOld[i].y = pTemp[i].y;

		//Rain effect
		/*p[i].y = threadIdx.x / 256.0*2;
		pOld[i].y = threadIdx.x / 256.0 * 2;*/
	}

}


static void display(void)
{

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	currentTime = glutGet(GLUT_ELAPSED_TIME);
	float deltaTime = (currentTime - previousTime)>16 ? 0.016 : (currentTime - previousTime)/1000.0;
	//printf(" %f ", deltaTime);

	previousTime = currentTime;

	cudaError_t cudaStatus;
	cudaStatus = calcVerletStep(p, pOld, pTemp, a, deltaTime, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}

	glBegin(GL_POINTS);
	for (unsigned int i = 0; i<arraySize; i++)
	{
		
		glColor4f(1.0f, 1.0f, 1.0f, 0.5f);
		glVertex3f(p[i].x, p[i].y, p[i].z );
	}
	glEnd();
	
	glutSwapBuffers();
	glutPostRedisplay();

}


void timer(int extra)
{
	glutPostRedisplay();
	glutTimerFunc(16, timer, 0);
}

int main(int argc, char **argv)
{
    

  
	initializePositions(p, pOld, arraySize);

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Verlet Integrator");

	glTranslatef(0.0f, -0.9f, 0.0f);
	glutDisplayFunc(display);
	//glutTimerFunc(0, timer, 0);
	glutMainLoop();


	/*
	cudaError_t cudaStatus;
	for (int i = 0; i < 1000; i++){

		// Add vectors in parallel.
		cudaStatus = calcVerletStep(p, pOld, pTemp, a, 0.016f, arraySize);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}
	}
	
	printf("Result: ");

	for (const float3 &position : p){
		print(position);
	}

	for (const float3 &position : pOld){
		print(position);
	}
	

	print(p[arraySize-1]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	*/
    return 0;
}

void initializePositions(float3 *p, float3 *pOld, int size){
	srand(time(0));
	for (int i = 0; i < size; i++){
		float xPos = (rand() % 2000 - 1000) / 1000.0;
		float yPos = (rand() % 100000 +50000) / 100000.0;
		float zPos = (rand() % 2000 - 1000) / 1000.0;
		p[i] = make_float3(xPos, yPos, zPos);
		pOld[i] = make_float3(xPos, yPos, zPos);
	}
}

void update(float timestep) {
	//Update the positions and velocities
	// xi+1 = xi + (xi - xi-1) + a * dt * dt


}

void print(float3 v){
	printf("(%f, %f, %f)", v.x, v.y, v.z);
}

//Round a / b to nearest higher integer value
int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
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


	int numThreads = 256;
	int numBlocks = iDivUp(size, numThreads); 
    // Launch a kernel on the GPU with one thread for each element.
	verletKernel <<<numBlocks, numThreads >>>(dev_p, dev_pOld, dev_pTemp, a, dt);

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
