#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

extern "C" {
	#include "camera.h"
}

// OpenGL Graphics includes
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors
(cudaError err, const char *file, const int line )
{
	if (err!=cudaSuccess)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
		file, line, (int)err, cudaGetErrorString(err));
		system("pause");
		exit(1);
	}
}

#define SCREEN_X 800
#define SCREEN_Y 800
#define FPS_UPDATE 200 
#define TITLE "N-Body"

#define MASSMAX 5.0f

#define CPU_MODE 1
#define GPU_MODE 2

#define BLOCK_SIZE 32



/* Globals */

int mode = CPU_MODE;
int frame=0;
int timebase=0;

float4 *pos=NULL, *vel=NULL;
float* mass=NULL;
float4 *d_pos=NULL, *d_vel=NULL;
float* d_mass=NULL;
int nbBodies = 1024;

float4* randomArrayFloat4(int n, float d)
{
	float4* a = (float4*)malloc(n*sizeof(float4));
	float x, y, z, r;
	int i;
	for (i=0;i<n;i++)
	{
		x = (2*((rand()%1000)/1000.0f)-1);
		y = (2*((rand()%1000)/1000.0f)-1);
		z = (2*((rand()%1000)/1000.0f)-1);
		r = (rand()%1000)/1000.0f/sqrt(x*x+y*y+z*z);

		a[i].x = r*d*x;
		a[i].y = r*d*y;
		a[i].z = r*d*z;
		a[i].w = 1.0f; // must be 1.0
	}
	return a;
}

float* randomArrayFloat(int n, float min, float max)
{
	float* a = (float*)malloc(n*sizeof(float));
	int i;
	for (i=0;i<n;i++)
		a[i] = min+(max-min)*((rand()%1000)/1000.0f);
	return a;
}

void initCPU()
{
	pos = randomArrayFloat4(nbBodies,1.0f);
	vel = randomArrayFloat4(nbBodies,0.0001f);
	mass = randomArrayFloat(nbBodies,1.0f, MASSMAX);
}

void initGPU()
{
	pos = randomArrayFloat4(nbBodies,1.0f);
	vel = randomArrayFloat4(nbBodies,0.0001f);
	mass = randomArrayFloat(nbBodies,1.0f, MASSMAX);
	
	checkCudaErrors(cudaMalloc(&d_pos,nbBodies*sizeof(float4)));
	checkCudaErrors(cudaMalloc(&d_vel,nbBodies*sizeof(float4)));
	checkCudaErrors(cudaMalloc(&d_mass,nbBodies*sizeof(float)));
	
	checkCudaErrors(cudaMemcpy(d_pos, pos, nbBodies*sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vel, vel, nbBodies*sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mass, mass, nbBodies*sizeof(float), cudaMemcpyHostToDevice));
}

void cleanCPU()
{
	if (pos) { free(pos);	pos = NULL; }
	if (vel) { free(vel);	vel = NULL; }
	if (mass) {	free(mass);	mass = NULL; }
}

void cleanGPU()
{
	if (pos) { free(pos);	pos = NULL; }
	if (vel) { free(vel);	vel = NULL; }
	if (mass) {	free(mass);	mass = NULL; }
	
	checkCudaErrors(cudaFree(d_pos));
	checkCudaErrors(cudaFree(d_vel));
	checkCudaErrors(cudaFree(d_mass));
}


void exampleCPU()
{
	// your N-body algorithm here!

	int i, j;

	for (i = 0; i<nbBodies; i++)
	{
		for (j = 0; j<nbBodies; j++)
		{
			float4 r;
			r.x = pos[j].x - pos[i].x;
			r.y = pos[j].y - pos[i].y;
			r.z = pos[j].z - pos[i].z;
			float d = r.x*r.x + r.y*r.y + r.z*r.z + 0.1;
			float f = 0.0000001f*mass[j] / sqrtf(d*d*d);
			vel[i].x+=r.x*f;
			vel[i].y+=r.y*f;
			vel[i].z+=r.z*f;
		}
	}
	for (i = 0; i<nbBodies; i++)
	{
		pos[i].x += vel[i].x;
		pos[i].y += vel[i].y;
		pos[i].z += vel[i].z;
	}
}

__global__ void exampleGPU(float4* pos,float4* vel,float* mass,int nbBodies)
{
	__shared__ float4 shPos[BLOCK_SIZE];
	__shared__ float shMass[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i>nbBodies)
		return;
	float4 mpos = pos[i];
	float mmass = mass[i];
	float4 macc = make_float4(0.0,0.0,0.0,0.0);

	
	for(int bi = 0;bi<blockDim.x;bi++)
	{
		int id = bi+BLOCK_SIZE+threadIdx.x;
		if(id<nbBodies)
		{
			shPos[threadIdx.x] = pos[id];
			shMass[threadIdx.x] = mass[id];
		}
		__syncthreads();
		for(int ti = 0;ti<BLOCK_SIZE;ti++){
			if(bi+BLOCK_SIZE+ti>=nbBodies)
				break;
			float4 r;
			r.x = shPos[ti].x - mpos.x;
			r.y = shPos[ti].y - mpos.y;
			r.z = shPos[ti].z - mpos.z;
			float d = r.x*r.x + r.y*r.y + r.z*r.z + 0.1;
			float f = 0.0000001f*shMass[threadIdx.x] / sqrtf(d*d*d);
			macc.x+=r.x*f;
			macc.y+=r.y*f;
			macc.z+=r.z*f;
		}
		__syncthreads();
	}
	vel[i].x+=macc.x;
	vel[i].y+=macc.y;
	vel[i].z+=macc.z;
	pos[i].x+=vel[i].x;
	pos[i].y+=vel[i].y;
	pos[i].z+=vel[i].z;
}

void calcNbodies() {
	frame++;
	int timecur = glutGet(GLUT_ELAPSED_TIME);

	if (timecur - timebase > FPS_UPDATE) {
		char t[200];
		char* m="";
		switch (mode)
		{
			case CPU_MODE: m = "CPU"; break;
			case GPU_MODE: m = "GPU"; break;
		}
		sprintf(t,"%s (mode: %s, bodies: %i, FPS: %.2f)",TITLE,m,nbBodies,frame*1000/(float)(timecur-timebase));
		glutSetWindowTitle(t);
	 	timebase = timecur;
		frame = 0;
	}

	switch (mode)
	{
		case CPU_MODE: exampleCPU(); break;
		case GPU_MODE:
		{
			dim3 nbBlock((nbBodies + BLOCK_SIZE - 1)/BLOCK_SIZE+1);
			dim3 nbThread(BLOCK_SIZE);
			exampleGPU<<<nbBlock,nbThread>>>(d_pos,d_vel,d_mass,nbBodies);
			checkCudaErrors(cudaGetLastError());
			cudaDeviceSynchronize();
			checkCudaErrors(cudaMemcpy(pos, d_pos, nbBodies*sizeof(float4), cudaMemcpyDeviceToHost));
			break;
		}
	}
}

void idleNbodies()
{
	glutPostRedisplay();
}


void renderNbodies(void)
{	
	calcNbodies();
	cameraApply();

	glClear(GL_COLOR_BUFFER_BIT);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(4, GL_FLOAT, 0, pos);
	glDrawArrays(GL_POINTS, 0, nbBodies);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	glutSwapBuffers();
}

void clean()
{
	switch (mode)
	{
		case CPU_MODE: cleanCPU(); break;
		case GPU_MODE: cleanGPU(); break;
	}
}

void init()
{
	switch (mode)
	{
		case CPU_MODE: initCPU(); break;
		case GPU_MODE: initGPU(); break;
	}

}

void toggleMode(int m)
{
	clean();
	mode = m;
	init();
}

void processNormalKeys(unsigned char key, int x, int y) {
	if (key == 27) exit(0);
	else if (key=='1') toggleMode(CPU_MODE);
	else if (key=='2') toggleMode(GPU_MODE);
}

void processSpecialKeys(int key, int x, int y) {

	switch(key) {
		case GLUT_KEY_UP: 
			if (nbBodies <16) nbBodies ++; 
			else if (nbBodies <128) nbBodies += 16; 
			else if (nbBodies <1024) nbBodies += 128;
			else if (nbBodies <1024*16) nbBodies += 512;
			else nbBodies += 1024*16;
			toggleMode(mode);
			break;
		case GLUT_KEY_DOWN: 
			if (nbBodies>1024*16) nbBodies -= 1024*16;
			else if (nbBodies>1024) nbBodies -= 512;
			else if (nbBodies>128) nbBodies -= 128;
			else if (nbBodies>16) nbBodies -= 16;
			else if (nbBodies>1) nbBodies--; 
			toggleMode(mode);
			break;
	}
	
}

void initGL(int argc, char **argv)
{
	// init GLUT and create window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(0,0);
	glutInitWindowSize(SCREEN_X,SCREEN_Y);
	glutCreateWindow(TITLE);
	glClearColor(0.0,0.0,0.0,0.0);
	glColor4f(1.0,1.0,1.0,1.0);
	glDisable(GL_DEPTH_TEST);
	glPointSize(1.0f);
}


int main(int argc, char **argv) {

	srand(time(NULL));
	initGL(argc, argv);


	toggleMode(CPU_MODE);

	glutDisplayFunc(renderNbodies);
	glutIdleFunc(idleNbodies);
	glutMouseFunc(trackballMouseFunction);
	glutMotionFunc(trackballMotionFunction);
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);

	glutMainLoop(); 

	clean();

	return 1;
}
