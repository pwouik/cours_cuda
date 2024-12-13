#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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
#include <cfloat>

extern "C" {
#include "camera.h"
}

#define SCREEN_X 800
#define SCREEN_Y 800
#define FPS_UPDATE 200 
#define TITLE "K-means"

enum {
	CPU_MODE = 1,
	GPU_MODE1,
	GPU_MODE2
};


int nbPoints = 2 * 1024 * 1024 ;
#define CLUSTERS 128
#define BLOCK_SIZE 32

/* Globals */

int mode = CPU_MODE;
int frame = 0;
int timebase = 0;

float4 *points = NULL, *centroids = NULL, *newCentroids = NULL, *pointColors = NULL, *centroidColors = NULL;
unsigned int* pointLabel = NULL, *newCentroidSize = NULL;

float4 *d_points = NULL, *d_centroids = NULL, *d_pointColors = NULL, *d_centroidColors = NULL;
unsigned int* d_pointLabel = NULL, *d_newCentroidSize = NULL;

float4* createRandomData(int n, float d)
{
	// create artificial clusters
	int nbClusters = CLUSTERS;
	int i = 0;
	float x, y, z, r;
	float4* c = (float4*)malloc(nbClusters*sizeof(float4));
	float* s = (float*)malloc(nbClusters*sizeof(float));
	for (i = 0; i<nbClusters; i++)
	{
		x = (2 * ((rand() % 1000) / 1000.0f) - 1);
		y = (2 * ((rand() % 1000) / 1000.0f) - 1);
		z = (2 * ((rand() % 1000) / 1000.0f) - 1);
		r = powf(3 * (rand() % 1000) / 1000.0f, 4);

		c[i].x = r*d*x;
		c[i].y = r*d*y;
		c[i].z = r*d*z;
		c[i].w = 1.0f; // must be 1.0

		s[i] = (rand() % 1000) / 1000.0f + 0.5f;
	}

	float4* a = (float4*)malloc(n*sizeof(float4));
	for (i = 0; i<n; i++)
	{
		int cl = rand() % CLUSTERS;
		x = (2 * ((rand() % 1000) / 1000.0f) - 1);
		y = (2 * ((rand() % 1000) / 1000.0f) - 1);
		z = (2 * ((rand() % 1000) / 1000.0f) - 1);
		r = powf(2 * (rand() % 1000) / 1000.0f / sqrt(x*x + y*y + z*z), 2.5);

		a[i].x = c[cl].x + s[cl] * s[cl] * r*d*x;
		a[i].y = c[cl].y + s[cl] * s[cl] * r*d*y;
		a[i].z = c[cl].z + s[cl] * s[cl] * r*d*z;
		a[i].w = 1.0f; // must be 1.0
	}
	free(c);
	free(s);
	return a;
}

float4 randomColor()
{
	float4 color;
	color.x = (rand() % 1000) / 1000.0f;
	color.y = (rand() % 1000) / 1000.0f;
	color.z = (rand() % 1000) / 1000.0f;
	color.w = 1.0f;
	return color;
}


void initCPU()
{
	newCentroids = (float4*)malloc(CLUSTERS*sizeof(float4));
	newCentroidSize = (unsigned int*)malloc(CLUSTERS*sizeof(unsigned int));

	points = createRandomData(nbPoints, 1.0f);
	pointColors = (float4*)malloc(nbPoints*sizeof(float4));
	pointLabel = (unsigned int*)malloc(nbPoints*sizeof(unsigned int));

	centroids = (float4*)malloc(CLUSTERS*sizeof(float4));
	centroidColors = (float4*)malloc(CLUSTERS*sizeof(float4));
	int i;
	for (i = 0; i<CLUSTERS; i++)
	{
		centroids[i] = points[i];  // Forgy method initialisation
		centroidColors[i] = randomColor();
	}
	for (i = 0; i<nbPoints; i++)
	{
		pointLabel[i] = 0;
	}

}

void cleanCPU()
{
	if (points) { free(points);	points = NULL; }
	if (pointLabel) { free(pointLabel);	pointLabel = NULL; }
	if (pointColors) { free(pointColors); pointColors = NULL; }
	if (centroids) { free(centroids); centroids = NULL; }
	if (centroidColors) { free(centroidColors); centroidColors = NULL; }
	if (newCentroidSize) { free(newCentroidSize); newCentroidSize = NULL; }
	if (newCentroids) { free(newCentroids); newCentroids = NULL; }
}

void initGPU()
{
	
	cudaMallocHost(&newCentroids, CLUSTERS*sizeof(float4));
	cudaMallocHost(&newCentroidSize,CLUSTERS*sizeof(unsigned int));

	points = createRandomData(nbPoints, 1.0f);
	cudaMallocHost(&pointColors,nbPoints*sizeof(float4));
	cudaMallocHost(&pointLabel,nbPoints*sizeof(unsigned int));

	cudaMallocHost(&centroids,CLUSTERS*sizeof(float4));
	cudaMallocHost(&centroidColors,CLUSTERS*sizeof(float4));
	int i;
	for (i = 0; i<CLUSTERS; i++)
	{
		centroids[i] = points[i];  // Forgy method initialisation
		centroidColors[i] = randomColor();
	}
	for (i = 0; i<nbPoints; i++)
	{
		pointLabel[i] = 0;
	}
	cudaMalloc(&d_points,nbPoints*sizeof(float4));
	cudaMalloc(&d_pointColors,nbPoints*sizeof(float4));
	cudaMalloc(&d_pointLabel,nbPoints*sizeof(unsigned int));
	cudaMalloc(&d_centroids,CLUSTERS*sizeof(float4));
	cudaMalloc(&d_newCentroidSize,CLUSTERS*sizeof(unsigned int));
	cudaMalloc(&d_centroidColors,CLUSTERS*sizeof(float4));
	cudaMemcpy(d_points,points,nbPoints*sizeof(float4),cudaMemcpyHostToDevice);
	cudaMemcpy(d_centroids,centroids,CLUSTERS*sizeof(float4),cudaMemcpyHostToDevice);
	cudaMemcpy(d_centroidColors,centroidColors,CLUSTERS*sizeof(float4),cudaMemcpyHostToDevice);

}

void cleanGPU()
{
	if (points) { free(points);	points = NULL; }
	if (pointLabel) { cudaFreeHost(pointLabel);	pointLabel = NULL; }
	if (pointColors) { cudaFreeHost(pointColors); pointColors = NULL; }
	if (centroids) { cudaFreeHost(centroids); centroids = NULL; }
	if (centroidColors) { cudaFreeHost(centroidColors); centroidColors = NULL; }
	if (newCentroidSize) { cudaFreeHost(newCentroidSize); newCentroidSize = NULL; }
	if (newCentroids) { cudaFreeHost(newCentroids); newCentroids = NULL; }

	
	if (d_points) { cudaFree(d_points);	d_points = NULL; }
	if (d_pointLabel) { cudaFree(d_pointLabel);	d_pointLabel = NULL; }
	if (d_pointColors) { cudaFree(d_pointColors); d_pointColors = NULL; }
	if (d_centroids) { cudaFree(d_centroids); d_centroids = NULL; }
	if (d_centroidColors) { cudaFree(d_centroidColors); d_centroidColors = NULL; }
}

void assignmentCPU(){
	for(unsigned int i = 0;i<nbPoints;i++){
		float dmin = 1e99;
		unsigned int n = 0;
		for(unsigned int j = 0;j<CLUSTERS;j++){
			float4 dist;
			dist.x=points[i].x-centroids[j].x;
			dist.y=points[i].y-centroids[j].y;
			dist.z=points[i].z-centroids[j].z;
			float distance = sqrtf(dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
			if (distance < dmin)
			{
				dmin = distance;
				n = j;
			}
		}
		pointLabel[i] = n; // point i assigned to cluster n
		pointColors[i] = centroidColors[n%CLUSTERS];
	}
}
void reduceCPU(){
	for(unsigned int j = 0;j<CLUSTERS;j++){
		newCentroids[j].x = 0.0f;
		newCentroids[j].y = 0.0f;
		newCentroids[j].z = 0.0f;
		newCentroidSize[j] = 0;
	}
	for(unsigned int i = 0;i<nbPoints;i++){
		newCentroids[pointLabel[i]].x += points[i].x;
		newCentroids[pointLabel[i]].y += points[i].y;
		newCentroids[pointLabel[i]].z += points[i].z;
		newCentroidSize[pointLabel[i]]++;
	}
	for(unsigned int j = 0;j<CLUSTERS;j++){
		centroids[j].x = newCentroids[j].x / (float)newCentroidSize[j];
		centroids[j].y = newCentroids[j].y / (float)newCentroidSize[j];
		centroids[j].z = newCentroids[j].z / (float)newCentroidSize[j];
	}
}

void exampleCPU()
{
	assignmentCPU();
	reduceCPU();
}

__global__ void assignmentGPU(int nbPoints,float4* points,float4* centroids,unsigned int* pointLabel,float4* pointColors,float4* centroidColors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i>=nbPoints){
		return;
	}
	float dmin = FLT_MAX;
	unsigned int n = 0;
	for(unsigned int j = 0;j<CLUSTERS;j++){
		float4 dist;
		dist.x=points[i].x-centroids[j].x;
		dist.y=points[i].y-centroids[j].y;
		dist.z=points[i].z-centroids[j].z;
		float distance = sqrtf(dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (distance < dmin)
		{
			dmin = distance;
			n = j;
		}
	}
	pointLabel[i] = n; // point i assigned to cluster n
	pointColors[i] = centroidColors[n%CLUSTERS];
}

__global__ void reduceGPU(int nbPoints,float4* points,float4* centroids,unsigned int* pointLabel,unsigned int* newCentroidSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i>=nbPoints){
		return;
	}
	unsigned int j = pointLabel[i];
	atomicAdd(&centroids[j].x,points[i].x);
	atomicAdd(&centroids[j].y,points[i].y);
	atomicAdd(&centroids[j].z,points[i].z);
	atomicAdd(&newCentroidSize[j],1);
}

__global__ void normalizeGPU(float4* centroids,unsigned int* newCentroidSize)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j>=CLUSTERS){
		return;
	}
	centroids[j].x/=newCentroidSize[j];
	centroids[j].y/=newCentroidSize[j];
	centroids[j].z/=newCentroidSize[j];
}

void exampleGPU1(){
	cudaMemcpy(d_centroids,centroids,CLUSTERS*sizeof(float4),cudaMemcpyHostToDevice);
	assignmentGPU<<<(nbPoints-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(nbPoints,d_points,d_centroids,d_pointLabel,d_pointColors,d_centroidColors);
	cudaDeviceSynchronize();
	cudaMemcpy(pointLabel,d_pointLabel,nbPoints*sizeof(unsigned int),cudaMemcpyDeviceToHost);
	cudaMemcpy(pointColors,d_pointColors,nbPoints*sizeof(float4),cudaMemcpyDeviceToHost);
	reduceCPU();
}


void exampleGPU2(){
	cudaDeviceSynchronize();
	assignmentGPU<<<(nbPoints-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(nbPoints,d_points,d_centroids,d_pointLabel,d_pointColors,d_centroidColors);
	cudaDeviceSynchronize();
	cudaMemset(d_newCentroidSize,0,CLUSTERS*sizeof(unsigned int));
	reduceGPU<<<(nbPoints-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(nbPoints,d_points,d_centroids,d_pointLabel,d_newCentroidSize);
	normalizeGPU<<<(CLUSTERS-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(d_centroids,d_newCentroidSize);
	cudaDeviceSynchronize();
	cudaMemcpy(centroids,d_centroids,CLUSTERS*sizeof(float4),cudaMemcpyDeviceToHost);
	cudaMemcpy(pointColors,d_pointColors,nbPoints*sizeof(float4),cudaMemcpyDeviceToHost);
	cudaMemcpy(points,d_points,nbPoints*sizeof(float4),cudaMemcpyDeviceToHost);
}

void calcClusters() {

	frame++;
	int timecur = glutGet(GLUT_ELAPSED_TIME);

	if (timecur - timebase > FPS_UPDATE) {
		char t[200];
		char* m = "";
		switch (mode)
		{
		case CPU_MODE: m = "CPU"; break;
		case GPU_MODE1: m = "GPU"; break;
		}
		sprintf(t, "%s: %s, %i points, %.2f FPS", TITLE, m, nbPoints, frame * 1000 / (float)(timecur - timebase));
		glutSetWindowTitle(t);
		timebase = timecur;
		frame = 0;
	}

	switch (mode)
	{
	case CPU_MODE: exampleCPU(); break;
	case GPU_MODE1: exampleGPU1(); break;
	case GPU_MODE2: exampleGPU2(); break;
	}
}

void idleKmeans()
{
	glutPostRedisplay();
}


void renderKmeans(void)
{

	calcClusters();
	cameraApply();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPointSize(1.0f);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glVertexPointer(4, GL_FLOAT, 0, points);
	glColorPointer(4, GL_FLOAT, 0, pointColors);
	glDrawArrays(GL_POINTS, 0, nbPoints);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	glPointSize(5.0f);
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(4, GL_FLOAT, 0, centroids);
	glDrawArrays(GL_POINTS, 0, CLUSTERS);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
}

void clean()
{
	switch (mode)
	{
	case CPU_MODE: cleanCPU(); break;
	case GPU_MODE1: cleanGPU(); break;
	case GPU_MODE2: cleanGPU(); break;
	}
}

void init()
{
	
	srand(456);
	switch (mode)
	{
	case CPU_MODE: initCPU(); break;
	case GPU_MODE1: initGPU(); break;
	case GPU_MODE2: initGPU(); break;
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
	else if (key == '1') toggleMode(CPU_MODE);
	else if (key == '2') toggleMode(GPU_MODE1);
	else if (key == '3') toggleMode(GPU_MODE2);
}

void processSpecialKeys(int key, int x, int y) {

	switch (key) {
	case GLUT_KEY_UP:
		nbPoints *= 2;
		toggleMode(mode);
		break;
	case GLUT_KEY_DOWN:
		if (nbPoints>2 * CLUSTERS) nbPoints /= 2;
		toggleMode(mode);
		break;
	}
}

void initGL(int argc, char **argv)
{
	// init GLUT and create window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(SCREEN_X, SCREEN_Y);
	glutCreateWindow(TITLE);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glColor4f(1.0, 1.0, 1.0, 1.0);
	glEnable(GL_DEPTH_TEST);
}


int main(int argc, char **argv) {

	srand(time(NULL));
	initGL(argc, argv);


	toggleMode(CPU_MODE);

	glutDisplayFunc(renderKmeans);
	glutIdleFunc(idleKmeans);
	glutMouseFunc(trackballMouseFunction);
	glutMotionFunc(trackballMotionFunction);
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);

	glutMainLoop();

	clean();

	return 1;
}
