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
#define TITLE "Julia Fractals"

#define CPU_MODE 1
#define GPU_MODE 2

GLuint imageTex;
GLuint imageBuffer;
float* debug;

/* Globals */
float scale = 0.003f;
float mx, my;
int mode = GPU_MODE;
int frame = 0;
int timebase = 0;
int max_iter = 1000;
int block_size = 16;

float4 *pixels;
float4 *d_pixels;

void initCPU()
{
	pixels = (float4*)malloc(SCREEN_X*SCREEN_Y*sizeof(float4));
}

void cleanCPU()
{
	free(pixels);
}

void initGPU()
{
	checkCudaErrors(cudaMalloc(&d_pixels,SCREEN_X*SCREEN_Y*sizeof(float4)));
	cudaMallocHost(&pixels,SCREEN_X*SCREEN_Y*sizeof(float4));
}

void cleanGPU()
{
	checkCudaErrors(cudaFree(d_pixels));
	cudaFreeHost(pixels);
}

void exampleCPU()
{
	int i, j;
	for (i = 0; i<SCREEN_Y; i++)
	for (j = 0; j<SCREEN_X; j++)
	{
		float x = (float)(scale*(j - SCREEN_X / 2));
		float y = (float)(scale*(i - SCREEN_Y / 2));
		float4* p = pixels + (i*SCREEN_X + j);
        float z_x = x;
        float z_y = y;
        float c_x = mx;
        float c_y = my;
        int i=0;
        while (z_x*z_x + z_y*z_y<4 && i<max_iter) {
            float tmp = z_x;
            z_x = z_x*z_x-z_y*z_y+c_x;
            z_y = 2*tmp*z_y+c_y;
            i++;
        }
		float c1 = (float)i/(float)max_iter;
        z_x = 0;
        z_y = 0;
        c_x = x;
        c_y = y;
    	i=0;
        while (z_x*z_x + z_y*z_y<4 && i<max_iter) {
            float tmp = z_x;
            z_x = z_x*z_x-z_y*z_y+c_x;
            z_y = 2*tmp*z_y+c_y;
            i++;
        }
		float c2 = (float)i/(float)max_iter;
		
		p->x = c1;
		p->y = c2;
		p->z = c1;
		p->w = 1.0f;
	}
}

__global__ void exampleGPU(float4* d_pixels,int sx,int sy,float scale,float mx,float my,int max_iter)
{
	
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if(idx>=SCREEN_X || idy>=SCREEN_Y )
		return;
	float x = (float)(scale*(idx - sx / 2));
	float y = (float)(scale*(idy - sy / 2));
	float4* p = d_pixels + (idy*sx + idx);
	float z_x = x;
	float z_y = y;
	float c_x = mx;
	float c_y = my;
	int i=0;
	while (z_x*z_x + z_y*z_y<4 && i<max_iter) {
		float tmp = z_x;
		z_x = z_x*z_x-z_y*z_y+c_x;
		z_y = 2*tmp*z_y+c_y;
		i++;
	}
	float c1 = (float)i/(float)max_iter;
	z_x = 0;
	z_y = 0;
	c_x = x;
	c_y = y;
	i=0;
	while (z_x*z_x + z_y*z_y<4 && i<max_iter) {
		float tmp = z_x;
		z_x = z_x*z_x-z_y*z_y+c_x;
		z_y = 2*tmp*z_y+c_y;
		i++;
	}
	float c2 = (float)i/(float)max_iter;
	float c = 0.5 + (c2*2.0-1.0)*(c1-0.5);
	p->x = c;
	p->y = c;
	p->z = c;
	p->w = 1.0f;
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
		sprintf(t, "%s:  %s, %.2f FPS", TITLE, m, frame * 1000 / (float)(timecur - timebase));
		glutSetWindowTitle(t);
		timebase = timecur;
		frame = 0;
	}

	switch (mode)
	{
	case CPU_MODE: exampleCPU(); break;
	case GPU_MODE:
		dim3 nbBlock;
		nbBlock.x = (SCREEN_X + block_size - 1)/block_size;
		nbBlock.y = (SCREEN_X + block_size - 1)/block_size+1;
		nbBlock.z = 1;
		dim3 nbThread;
		nbThread.x = block_size;
		nbThread.y = block_size;
		nbThread.z = 1;
		exampleGPU<<<nbBlock,nbThread>>>(d_pixels,SCREEN_X,SCREEN_Y,scale,mx,my,max_iter);
		checkCudaErrors(cudaGetLastError());
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy(pixels, d_pixels, SCREEN_X*SCREEN_Y*sizeof(float4), cudaMemcpyDeviceToHost));
		break;
	}
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
