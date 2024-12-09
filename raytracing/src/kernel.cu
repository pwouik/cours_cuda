#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <stdio.h>

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


#define SCREEN_X 1024
#define SCREEN_Y 768
#define FPS_UPDATE 500
#define TITLE "Raytracing"

#define CPU_MODE 1
#define GPU_MODE 2


#define MAX_STREAMS 8
int num_streams = 1;
cudaStream_t streams[MAX_STREAMS];

GLuint imageTex;
GLuint imageBuffer;
float* debug;

/* Globals */
float scale = 0.003f;
float mx, my;
int mode = GPU_MODE;
int frame = 0;
int timebase = 0;
int block_size = 16;

float4 *pixels;
float4 *d_pixels;

#define INF 2e10f
struct Sphere {
	float r,g,b;
	float radius;
	float x,y,z;
	__host__ __device__ float hit(float cx, float cy, float *sh) {
		float dx = cx - x;
		float dy = cy - y;
		float dz2 = radius*radius - dx*dx - dy*dy;
		if (dz2>0) {
			float dz = sqrtf(dz2);
			*sh = dz / radius;
			return dz + z;
		}
		return -INF;
	}
};
#define MAX_SPHERES 1000
int nSpheres = 10;
__constant__ Sphere c_spheres[MAX_SPHERES];
Sphere *spheres;
Sphere *d_spheres;

float randf(float a,float b){
	return (float)rand()/RAND_MAX*(b-a)+a;
}

void initCPU()
{

	spheres = (Sphere*)malloc(MAX_SPHERES*sizeof(Sphere));
	for(int i=0;i<MAX_SPHERES;i++){
		spheres[i].x = randf(-5.0, 5.0);
		spheres[i].y = randf(-5.0, 5.0);
		spheres[i].z = randf(0.0, 1.0);
		spheres[i].r = randf(0.0, 1.0);
		spheres[i].g = randf(0.0, 1.0);
		spheres[i].b = randf(0.0, 1.0);
		spheres[i].radius = randf(0.5, 2.0);
	}
	pixels = (float4*)malloc(SCREEN_X*SCREEN_Y*sizeof(float4));
}

void cleanCPU()
{
	free(pixels);
	free(spheres);
}
void initGPU()
{
    for (int i = 0; i < MAX_STREAMS; i++)
    	cudaStreamCreate(&streams[i]);
	cudaMallocHost(&spheres,MAX_SPHERES*sizeof(Sphere));
	for(int i=0;i<MAX_SPHERES;i++){
		spheres[i].x = randf(-5.0, 5.0);
		spheres[i].y = randf(-5.0, 5.0);
		spheres[i].z = randf(0.0, 1.0);
		spheres[i].r = randf(0.0, 1.0);
		spheres[i].g = randf(0.0, 1.0);
		spheres[i].b = randf(0.0, 1.0);
		spheres[i].radius = randf(0.1, 0.5);
	}
	//checkCudaErrors(cudaMalloc(&d_spheres,nSpheres*sizeof(Sphere)));
	//checkCudaErrors(cudaMemcpy(d_spheres, spheres, nSpheres*sizeof(Sphere), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpyToSymbol(c_spheres, spheres, MAX_SPHERES*sizeof(Sphere)));
	checkCudaErrors(cudaMalloc(&d_pixels,SCREEN_X*SCREEN_Y*sizeof(float4)));
	cudaMallocHost(&pixels,SCREEN_X*SCREEN_Y*sizeof(float4));
}

void cleanGPU()
{
	checkCudaErrors(cudaFree(d_pixels));
	//checkCudaErrors(cudaFree(d_spheres));
	cudaFreeHost(pixels);
	cudaFreeHost(spheres);
}

void exampleCPU()
{
	int i, j;
	for (i = 0; i<SCREEN_Y; i++)
	for (j = 0; j<SCREEN_X; j++)
	{
		float x = (float)(scale*(j - SCREEN_X / 2)) + mx;
		float y = (float)(scale*(i - SCREEN_Y / 2)) + my;
		float4* p = pixels + (i*SCREEN_X + j);
		p->x = 0.0;
		p->y = 0.0;
		p->z = 0.0;
		p->w = 1.0f;
		float z=-INF;
		for(int i=0;i<nSpheres;i++){
			float sh;
			float d =spheres[i].hit(x,y,&sh);
			if(d>z){
				z=d;
				p->x = spheres[i].r*sh;
				p->y = spheres[i].g*sh;
				p->z = spheres[i].b*sh;
			}
		}
	}
}

__global__ void exampleGPU(float4* d_pixels,int sx,int sy,int num_streams,int slice,float scale,float mx,float my,int nSpheres)
{
	
    int idx = blockIdx.x * blockDim.x + threadIdx.x+SCREEN_X*slice/num_streams;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if(idx>=SCREEN_X || idy>=SCREEN_Y )
		return;
	float x = (float)(scale*(idx - sx / 2)) + mx;
	float y = (float)(scale*(idy - sy / 2)) + my;
	float4* p = d_pixels + (idy*sx + idx);
	p->x = 0.0;
	p->y = 0.0;
	p->z = 0.0;
	p->w = 1.0f;
	float z=-INF;
	for(int i=0;i<nSpheres;i++){
		float sh;
		float d =c_spheres[i].hit(x,y,&sh);
		if(d>z){
			z=d;
			p->x = c_spheres[i].r*sh;
			p->y = c_spheres[i].g*sh;
			p->z = c_spheres[i].b*sh;
		}
	}
}

void calculate() {
	frame++;
	int timecur = glutGet(GLUT_ELAPSED_TIME);

	if (timecur - timebase > FPS_UPDATE) {
		char t[200];
		char* m = "";
		switch (mode)
		{
		case CPU_MODE: m = "CPU mode"; break;
		case GPU_MODE: m = "GPU mode"; break;
		}
		sprintf(t, "%s:  %s, %d Spheres,%d Streams,%.2f FPS", TITLE, m, nSpheres ,num_streams , frame * 1000 / (float)(timecur - timebase));
		glutSetWindowTitle(t);
		timebase = timecur;
		frame = 0;
	}
	auto start = std::chrono::high_resolution_clock::now();
	switch (mode)
	{
	case CPU_MODE: exampleCPU(); break;
	case GPU_MODE:
		dim3 nbBlock;
		nbBlock.x = (SCREEN_X/num_streams + block_size - 1)/block_size;
		nbBlock.y = (SCREEN_X + block_size - 1)/block_size+1;
		nbBlock.z = 1;
		dim3 nbThread;
		nbThread.x = block_size;
		nbThread.y = block_size;
		nbThread.z = 1;
    	for (int i = 0; i < num_streams; i++)
			exampleGPU<<<nbBlock,nbThread,0,streams[i]>>>(d_pixels,SCREEN_X,SCREEN_Y,num_streams,i,scale,mx,my,nSpheres);
		checkCudaErrors(cudaGetLastError());
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy(pixels, d_pixels, SCREEN_X*SCREEN_Y*sizeof(float4), cudaMemcpyDeviceToHost));
		break;
	}
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() << "µs\n";
}

void idle()
{
	glutPostRedisplay();
}


void render()
{
	calculate();
	switch (mode)
	{
	case CPU_MODE: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels); break;
	case GPU_MODE:
		glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels);
		break;
	}
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

void mouse(int button, int state, int x, int y)
{
	if (button <= 2)
	{
		mx = (float)(scale*(x - SCREEN_X / 2));
		my = -(float)(scale*(y - SCREEN_Y / 2));
	}
	// Wheel reports as button 3 (scroll up) and button 4 (scroll down)
	if (button == 3) scale /= 1.05f;
	else if (button == 4) scale *= 1.05f;
}

void mouseMotion(int x, int y)
{
	mx = (float)(scale*(x - SCREEN_X / 2));
	my = -(float)(scale*(y - SCREEN_Y / 2));
}

void processNormalKeys(unsigned char key, int x, int y) {

	if (key == 27) { clean(); exit(0); }
	else if (key == '1') toggleMode(CPU_MODE);
	else if (key == '2') toggleMode(GPU_MODE);
	else if (key == '7') nSpheres= std::min(nSpheres+1,MAX_SPHERES);
	else if (key == '4') nSpheres= std::max(nSpheres-1,0);
	else if (key == '8') num_streams= std::min(num_streams+1,MAX_STREAMS);
	else if (key == '5') num_streams= std::max(num_streams-1,1);
}

void processSpecialKeys(int key, int x, int y) {
	// other keys (F1, F2, arrows, home, etc.)
	switch (key) {
	case GLUT_KEY_UP: break;
	case GLUT_KEY_DOWN: break;
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
	glDisable(GL_DEPTH_TEST);

	// View Ortho
	// Sets up the OpenGL window so that (0,0) corresponds to the top left corner, 
	// and (SCREEN_X,SCREEN_Y) corresponds to the bottom right hand corner.  
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, SCREEN_X, SCREEN_Y, 0, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.375, 0.375, 0); // Displacement trick for exact pixelization
}



int main(int argc, char **argv) {

	initGL(argc, argv);

	init();

	glutDisplayFunc(render);
	glutIdleFunc(idle);
	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);

	// enter GLUT event processing cycle
	glutMainLoop();

	clean();

	return 1;
}
