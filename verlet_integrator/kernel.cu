
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "operators.h"
#include <gl/glew.h>
#include <GL/freeglut.h> 
#include <math.h>
#include <ctime>
#include "helper_cuda.h"         // helper functions for CUDA error check
#include "helper_cuda_gl.h"   
#include <cuda_gl_interop.h>
#include "utils.h";


cudaError_t calcVerletStep(float3 *p, float3 *pOld, float3 a, float dt, unsigned int size);
void print(float3 v);
void initializePositions(float3 *p, float3 *pOld, int size);
void runCuda(struct cudaGraphicsResource **vbo_resource, float deltaTime);

const unsigned int window_width = 1024;
const unsigned int window_height = 1024;


const unsigned int mesh_width = 512;
const unsigned int mesh_height = 512;

GLuint shaderProgram;


// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

const int arraySize = 250000;
float3 *p = new float3[arraySize];
float3 *pOld = new float3[arraySize];

float3 a = make_float3(0, -2.0f, 0);
int currentTime = 0;
int previousTime = 0;
int frameCount = 0;
float timeCount = 0;
float fps = 0;

//Cuda varables
float3 *dev_p = 0;
float3 *dev_pOld = 0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

__global__ void verletKernel(float4 *pos, float3 *p, float3 *pOld, float3 a, float dt)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	float3 pTemp = p[i];
    p[i] = p[i] + p[i] - pOld[i] + a* dt*dt;
	pOld[i] = pTemp;

	//Collision against floor
	if (p[i].y < 0){

		//Bouncing collision
		pTemp.y = p[i].y;
		p[i].y = pOld[i].y*0.5;
		pOld[i].y = pTemp.y;

		//Rain effect
		/*p[i].y = threadIdx.x / 256.0*2;
		pOld[i].y = threadIdx.x / 256.0 * 2;*/
	}

	// write output vertex
	pos[i] = make_float4(p[i].x, p[i].y, p[i].z, 1.0f);

}


static void display(void)
{


	currentTime = glutGet(GLUT_ELAPSED_TIME);
	float deltaTime = (currentTime - previousTime)>16 ? 0.016 : (currentTime - previousTime)/1000.0;
	timeCount += deltaTime;
	previousTime = currentTime;

	frameCount++;

	
	//printf(" %f ", deltaTime);

	if (timeCount >= 1){
		fps = frameCount/1.0 / timeCount;
		frameCount = 0;
		timeCount = 0;
	}
	char fpsText[256];
	sprintf(fpsText, "Verlet Integrator: %3.1f fps", fps);
	glutSetWindowTitle(fpsText);

	runCuda(&cuda_vbo_resource, deltaTime);


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	glUseProgram(shaderProgram);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(0.5, 0.5, 1.0);
	glDrawArrays(GL_POINTS, 0, arraySize);
	glDisableClientState(GL_VERTEX_ARRAY);

	glUseProgram(0);
	glutSwapBuffers();
	glutPostRedisplay();

}


void timer(int extra)
{
	glutPostRedisplay();
	glutTimerFunc(5, timer, 0);
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

void initializeCuda(float3 *p, float3 *pOld, int size){
	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));


	// Allocate GPU buffers for three vectors (two input, one output)    	
	checkCudaErrors(cudaMalloc((void**)&dev_p, size * sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&dev_pOld, size * sizeof(float3)));


	// Copy postition vectors from host memory to GPU buffers.

	checkCudaErrors(cudaMemcpy(dev_p, p, size * sizeof(float3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_pOld, pOld, size * sizeof(float3), cudaMemcpyHostToDevice));

}

void resetPostions(float3 *p, float3 *pOld, int size){
	initializePositions(p, pOld, size);
	// Copy postition vectors from host memory to GPU buffers.
	checkCudaErrors(cudaMemcpy(dev_p, p, size * sizeof(float3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_pOld, pOld, size * sizeof(float3), cudaMemcpyHostToDevice));
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags)
{
	//assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	//SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

void cleanup()
{

	if (vbo)
	{
		deleteVBO(&vbo, cuda_vbo_resource);
	}
	
	cudaFree(dev_p);
	cudaFree(dev_pOld);

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();
}



void launch_kernel(float4 *pos, unsigned int mesh_width,
	unsigned int mesh_height, float time)
{
	// execute the kernel
	/*dim3 block(8, 8, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	simple_vbo_kernel << < grid, block >> >(pos, mesh_width, mesh_height, time);*/


	int numThreads = 256;
	int numBlocks = iDivUp(arraySize, numThreads);
	// Launch a kernel on the GPU with one thread for each element.
	verletKernel <<<numBlocks, numThreads >>>(pos, dev_p, dev_pOld, a, time);
}


////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource, float deltaTime)
{
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
		*vbo_resource));
	
	launch_kernel(dptr, mesh_width, mesh_height, deltaTime);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case (27) :
#if defined(__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif
		break;

	case (32) :
		resetPostions(p, pOld, arraySize);
	}
}



int main(int argc, char **argv)
{



	initializePositions(p, pOld, arraySize);
	initializeCuda(p, pOld, arraySize);
	

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Verlet Integrator");

	//glTranslatef(0.0f, -0.9f, 0.0f);

	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());


	//Register callbacks
	glutDisplayFunc(display);
	//glutTimerFunc(0, timer, 0);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutCloseFunc(cleanup);


	glewInit();

	shaderProgram = loadShaders();

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);


	// create VBO
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	glutMainLoop();


	

	return 0;
}
