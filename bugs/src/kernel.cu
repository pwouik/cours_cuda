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

#define BUGS

#ifdef BUGS
#define RANGE 5
#define SURVIVELO 34
#define SURVIVEHI 58
#define BIRTHLO 34
#define BIRTHH 45
#endif

#ifdef LIFE
#define RANGE 1
#define SURVIVELO 3
#define SURVIVEHI 4
#define BIRTHLO 3
#define BIRTHH 3
#endif

GLuint imageTex;
GLuint imageBuffer;
float* debug;

int mode = GPU_MODE;
int frame_counter = 0;
int frame = 0;
int timebase = 0;
#define BLOCK_SIZE 16

float3 *pixels;
float3 *pixels1;
float3 *pixels2;
float3 *d_pixels1;
float3 *d_pixels2;
int radius = 10;


float randf(float a,float b){
	return (float)rand()/RAND_MAX*(b-a)+a;
}
__host__ __device__ int wrap(int n,int d){
	if(n<0){
		n+=d;
	}
	if(n>=d){
		n-=d;
	}
	return n;
}

void initCPU()
{
	pixels1 = (float3*)malloc(SCREEN_X*SCREEN_Y*sizeof(float3));
	pixels2 = (float3*)malloc(SCREEN_X*SCREEN_Y*sizeof(float3));
	for(size_t i=0;i<SCREEN_X*SCREEN_Y;i++){
		float3* p = &pixels1[i];
		if(rand()&1){
			p->x = 1.0;
			p->y = 1.0;
			p->z = 1.0;
		}
		else{
			p->x = 0.0;
			p->y = 0.0;
			p->z = 0.0;
		}
	}
}

void cleanCPU()
{
	free(pixels1);
	free(pixels2);
}
void initGPU()
{
	checkCudaErrors(cudaMalloc(&d_pixels1,SCREEN_X*SCREEN_Y*sizeof(float4)));
	checkCudaErrors(cudaMalloc(&d_pixels2,SCREEN_X*SCREEN_Y*sizeof(float4)));
	cudaMallocHost(&pixels,SCREEN_X*SCREEN_Y*sizeof(float4));
	for(size_t i=0;i<SCREEN_X*SCREEN_Y;i++){
		float3* p = &pixels[i];
		if(rand()&1){
			p->x = 1.0;
			p->y = 1.0;
			p->z = 1.0;
		}
		else{
			p->x = 0.0;
			p->y = 0.0;
			p->z = 0.0;
		}
	}
	checkCudaErrors(cudaMemcpy(d_pixels1, pixels, SCREEN_X*SCREEN_Y*sizeof(float3), cudaMemcpyHostToDevice));
}

void cleanGPU()
{
	checkCudaErrors(cudaFree(d_pixels1));
	checkCudaErrors(cudaFree(d_pixels2));
	cudaFreeHost(pixels);
}

void exampleCPU(float3* old_buffer,float3* new_buffer)
{
	int i, j;
	for (i = 0; i<SCREEN_Y; i++)
	for (j = 0; j<SCREEN_X; j++)
	{
		unsigned int c=0;
		for (int y = -RANGE; y<=RANGE; y++)
		for (int x = -RANGE; x<=RANGE; x++){
			 if(old_buffer[wrap(i+y,SCREEN_Y)*SCREEN_X+wrap(j+x,SCREEN_X)].x >0){
				c+=1;
			 }
		}
		float3* po = &old_buffer[i*SCREEN_X+j];
		float3* p = &new_buffer[i*SCREEN_X+j];
		if(c>=BIRTHLO && c<=BIRTHH){
			p->x = 1.0;
			p->y = 1.0;
			p->z = 1.0;
		}
		else if(c>=SURVIVELO && c<=SURVIVEHI){
			p->x = po->x;
			p->y = po->y;
			p->z = po->z;
		}
		else{
			p->x = 0.0;
			p->y = 0.0;
			p->z = 0.0;
		}
		
	}
}
#define TILE_WIDTH (2*RANGE + BLOCK_SIZE)

__global__ void exampleGPU(float3* old_buffer,float3* new_buffer)
{
	__shared__ float tile[TILE_WIDTH*TILE_WIDTH];
	int bx = blockIdx.x * blockDim.x;
	int by = blockIdx.y * blockDim.y;
    int j = bx + threadIdx.x;
    int i = by + threadIdx.y;
	if(j>=SCREEN_X || i>=SCREEN_Y )
		return;
	for (int y = threadIdx.y; y<TILE_WIDTH; y+=BLOCK_SIZE)
	for (int x = threadIdx.x; x<TILE_WIDTH; x+=BLOCK_SIZE){
		tile[y*TILE_WIDTH+x] = old_buffer[wrap(by-RANGE+y,SCREEN_Y)*SCREEN_X+wrap(bx-RANGE+x,SCREEN_X)].x;
	}

	__syncthreads();
	uint32_t c=0;
	for (int y = 0; y<=2*RANGE; y++)
	for (int x = 0; x<=2*RANGE; x++){
		int ty = y+threadIdx.y;
		int tx = x+threadIdx.x;
		if(tile[ty*TILE_WIDTH+tx] == 1.0){
			c+=1;
		}
	}
	float3* po = &old_buffer[i*SCREEN_X+j];
	float3* p = &new_buffer[i*SCREEN_X+j];
	if(c>=BIRTHLO && c<=BIRTHH && po->x==0.0){
		p->x = 1.0;
		p->y = 1.0;
		p->z = 1.0;
	}
	else if(c>=SURVIVELO && c<=SURVIVEHI){
		p->x = po->x;
		p->y = po->y;
		p->z = po->z;
	}
	else{
		p->x = 0.0;
		p->y = 0.0;
		p->z = 0.0;
	}
}

void calculate() {
	frame_counter++;
	int timecur = glutGet(GLUT_ELAPSED_TIME);

	if (timecur - timebase > FPS_UPDATE) {
		char t[200];
		char* m = "";
		switch (mode)
		{
			case CPU_MODE: m = "CPU mode"; break;
			case GPU_MODE: m = "GPU mode"; break;
		}
		sprintf(t, "%s:  %s,%.2f FPS", TITLE, m, frame_counter * 1000 / (float)(timecur - timebase));
		glutSetWindowTitle(t);
		timebase = timecur;
		frame_counter = 0;
	}
	frame++;
	auto start = std::chrono::high_resolution_clock::now();
	switch (mode)
	{
	case CPU_MODE:
		if(frame&1){
			exampleCPU(pixels1,pixels2);
			pixels=pixels2;
		}
		else{
			exampleCPU(pixels2,pixels1);
			pixels=pixels1;
		}
		break;
	case GPU_MODE:
		dim3 nbBlock;
		nbBlock.x = (SCREEN_X + BLOCK_SIZE - 1)/BLOCK_SIZE+1;
		nbBlock.y = (SCREEN_X + BLOCK_SIZE - 1)/BLOCK_SIZE+1;
		nbBlock.z = 1;
		dim3 nbThread;
		nbThread.x = BLOCK_SIZE;
		nbThread.y = BLOCK_SIZE;
		nbThread.z = 1;
		if(frame&1){
			exampleGPU<<<nbBlock,nbThread>>>(d_pixels1,d_pixels2);
			checkCudaErrors(cudaGetLastError());
			cudaDeviceSynchronize();
			checkCudaErrors(cudaMemcpy(pixels, d_pixels2, SCREEN_X*SCREEN_Y*sizeof(float3), cudaMemcpyDeviceToHost));
		}
		else{
			exampleGPU<<<nbBlock,nbThread>>>(d_pixels2,d_pixels1);
			checkCudaErrors(cudaGetLastError());
			cudaDeviceSynchronize();
			checkCudaErrors(cudaMemcpy(pixels, d_pixels1, SCREEN_X*SCREEN_Y*sizeof(float3), cudaMemcpyDeviceToHost));
		}
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
	case CPU_MODE: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGB, GL_FLOAT, pixels); break;
	case GPU_MODE:
		glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGB, GL_FLOAT, pixels);
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
	srand(time(NULL));
	frame=0;
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
}

void mouseMotion(int x, int y)
{
}

void processNormalKeys(unsigned char key, int x, int y) {

	if (key == 27) { clean(); exit(0); }
	else if (key == '1') toggleMode(CPU_MODE);
	else if (key == '2') toggleMode(GPU_MODE);
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
