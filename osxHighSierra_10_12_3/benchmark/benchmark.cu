#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <ctime>
#include <math.h>       /* pow, ceil */
#include <algorithm>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
//Windows has <direct.h>, POSIX systems have <unistd.h>
#include <unistd.h> /*To get the path to this script's directory*/
#include <sys/syslimits.h>

using namespace std;

const int C = 6144;
//const int b = 8;//To cause misses.
const int b = 0;//For no misses.
//const int N = C+b+1; //24580 bytes = 24KB + 4B in binary => first miss
const int N = 9*8 + C+(b*5);
//const int N = 6144; //24576 bytes = 24KB => no miss => C = 6144
//const int N = 6000; //24000 bytes = 24KB in decimal => no miss
const int s = 8;//s = 128*4 bytes

//Max iterations with stored array 
//indeces being ints 
//const int iterations = 0xc000/8 - 4;

//34 because 17*500 = 8500 => traverse array N once. 
//*2 for traversing twice to eliminate cold miss effect.
//The modulus so it will do N+1 when odd and N+0 when even.
//Just so the division always ends up being an even # div by 2.
const int iterations = ((N+(N%2))/s)*4;
//const int iterations = ((N+(N%2))/s);
//const int iterations = 6143;
//const int iterations = 768*2;
//Both threshold and tolerance
//dynamically allocated in main()
unsigned int threshold = 200; float tolerance = 0.3; //30% tolerance
const char *path = getcwd(NULL,0);

__global__ void bench_CacheAccess(unsigned int *CUDA_A, unsigned int device_tvalue[], unsigned int device_index[]) {
	//Placing variables in shared memory makes them
	//not interfere with the global memory cache and, hence, the experiment
	__shared__ unsigned int s_tvalue[iterations];
	__shared__ unsigned int s_index[iterations];
	//__shared__ unsigned int s_tvalue[iterations];
	//__shared__ unsigned int s_index[iterations];
	//__shared__ int j;
    int j;
	j = 0;
	for (int it = 0; it < iterations; it++) {
		clock_t start_time = clock();
		j = CUDA_A[j];
		//Store the element index
		//Also generates memory dependence on previous
		//instruction, so that clock() happens after the
		//array access above
		s_index[it] = j;
		clock_t end_time = clock();
		//store the access latency
		s_tvalue[it] = end_time - start_time;
	}
	//All threads in this block have to reach this point
	//before continuing execution.
	__syncthreads();

	//Transfer results from shared memory to global memory
	//Later we will memcpy() the device global memory to host
	for (int i = 0; i < iterations; i++) {
		device_index[i] = s_index[i];
		device_tvalue[i] = s_tvalue[i];
	}

}

int main()
{
	printf("Will go through [%d] iterations with array of size N = [%d].\n", iterations, N);
	FILE * file;
	unsigned int *A = new unsigned int[N]; //The array of size N to test the cache 
	unsigned int *host_tvalue = new unsigned int[iterations]; //Time values for memory accesses
	unsigned int *host_index = new unsigned int[iterations]; //Index array of the accesses to the array elements
	int hits = 0, misses = 0;
	//Initialize array
	for (int i = 0; i < N; i++) {
		A[i] = (i + s) % N;
	}
	//Initialize index and time value arrays
	for (int k = 0; k < iterations; k++) {
		host_tvalue[k] = 0;
		host_index[k] = 0;
	}

    cudaFuncSetCacheConfig(bench_CacheAccess, cudaFuncCachePreferL1);


	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return -1;
	}

	unsigned int *CUDA_A = 0; //When we allocate space for A on the GPU we assign it to this ptr, CUDA_A
	unsigned int *device_tvalue = 0; //Device variables needed to copy back to host.
	unsigned int *device_index = 0;


	//Places array into cache
	cudaStatus = cudaMalloc((void**)&CUDA_A, N * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cudaDeviceReset(); //Clear all allocations and exit
	}

	//Places array into cache
	cudaStatus = cudaMalloc((void**)&device_tvalue, iterations * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for the tvalues array!");
		cudaDeviceReset(); //Clear all allocations and exit
		return -1;
	}

	//Places array into cache
	cudaStatus = cudaMalloc((void**)&device_index, iterations * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for the index array!");
		cudaDeviceReset(); //Clear all allocations and exit
		return -1;
	}

	for (int a = 0; a < 10; a++){
		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(CUDA_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for the array!");
			cudaDeviceReset();
			return -1;
		}
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(device_index, host_index, iterations * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for the index array!");
		cudaDeviceReset();
		return -1;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(device_tvalue, host_tvalue, iterations * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for the tvalues array!");
		cudaDeviceReset();
	}

    // Classic P-chase benchmark.
	bench_CacheAccess<<<1,1>>>(CUDA_A, device_tvalue, device_index);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

	// cudadevicesynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudadevicesynchronize returned error code %d after launching kernel!\n", cudaStatus);
		return -1;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_tvalue, device_tvalue, iterations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! Could not retrieve tvalue from device.\n");
		return -1;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_index, device_index, iterations * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! Could not retrieve index from device.\n");
		return -1;
	}

	//Dynamically allocate threshold. First access is always a cold miss
	int delta = (int) (tolerance * host_tvalue[0]);
    //If I don't hardcode threshold, calculate it
	if (threshold == 0) threshold = host_tvalue[0] - delta;
	file = fopen("experiment_results.dat","w");
	if ((host_tvalue != NULL) && (host_index != NULL)) {
		for (int a = 0; a < iterations; a++) {
			if (host_tvalue[a] > threshold) misses++;
			else hits++;
		}
	}
	printf("%d hits\n%d misses\n", hits, misses);
	printf("threshold = %d\n", threshold);
    //printf("b = %d\n",((s*N)/(N-s*hits)));
	fprintf(file, "hits|misses\n");
	fprintf(file, "%d|%d\n", hits, misses);
	fprintf(file, "threshold=%d\n", threshold);
	fprintf(file, "arraySize=%d\n", N);
	fprintf(file, "stride=%d\n", s);
	fprintf(file, "numIterations=%d\n", iterations);
	fprintf(file,"arrayIndex|tvalue\n");
	for (int b = 0; b < iterations; b++) {
		//printf("host_index[%d] = %d\n",b,host_index[b]);
		//printf("host_tvalue[%d] = %d\n", b, host_tvalue[b]);
		fprintf(file,"%d|%d\n",host_index[b],host_tvalue[b]);
	}
	fprintf(file,"end\n");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	fclose(file);
	
	//Call python GUI script to show results
	printf("path = %s\n",path);
	//+7 because of "python\s"
	char cmd[PATH_MAX + 7];
	snprintf(cmd,sizeof(cmd),"python %s/../Python_Scripts/GUI.py", path);
	system(cmd);

    return 0;
}
