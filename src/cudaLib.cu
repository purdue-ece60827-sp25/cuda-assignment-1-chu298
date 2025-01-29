
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
	{
		y[i] = scale * x[i] + y[i];
	}
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";

	int size = vectorSize * sizeof(float);
	float *X_d, *Y_d, *Z_d;
	float *X, *Y, *Z;

	cudaMalloc((void**) &X_d, size);
	cudaMalloc((void**) &Y_d, size);
	cudaMalloc((void**) &Z_d, size);
	X = (float *) malloc(size);
	Y = (float *) malloc(size);
	Z = (float *) malloc(size);

	// init X and Y
	vectorInit(X, vectorSize);
	vectorInit(Y, vectorSize);

	cudaMemcpy(X_d, X, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Y_d, Y, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Z_d, Y, size, cudaMemcpyHostToDevice);

	float scale = rand() % 99999;
	saxpy_gpu<<<ceil(vectorSize/256.0), 256>>>(X_d, Z_d, scale, vectorSize);

	cudaMemcpy(Z, Z_d, size, cudaMemcpyDeviceToHost);

	cudaFree(X_d);
	cudaFree(Y_d);
	cudaFree(Z_d);

	// verify
	printf("Error count: %i\n", verifyVector(X, Y, Z, scale, vectorSize));

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}
