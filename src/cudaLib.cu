
#include "cudaLib.cuh"
#include "curand_kernel.h"

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
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	// Setup RNG
	curandState_t rng;
	curand_init(clock64(), id, 0, &rng);

	if (id < pSumSize)
	{
		for (uint64_t i = 0; i < sampleSize; i++)
		{
			float randX = curand_uniform(&rng);
			float randY = curand_uniform(&rng);

			if ( int(randX * randX + randY * randY) == 0 ) {
				pSums[id]++;
			}
		}
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < reduceSize)
	{
		for (uint64_t i = 0; i < pSumSize/reduceSize; i++)
		{
			totals[id] += pSums[id + i];
		}
	}
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
	
	std::cout << std::setprecision(10);
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

	generateThreadCount = 4096;
	sampleSize = 1000000000;
	reduceThreadCount = 64;
	reduceSize = 64;

	printf("Num Threads: %llu, Sample Size: %llu\n", generateThreadCount, sampleSize);
	printf("Num Reduced Threads: %llu, Reduce Size: %llu\n", reduceThreadCount, reduceSize);

	// Initialize stuff
	uint64_t *partialSums_d;
	uint64_t *partialSums;
	int size = generateThreadCount * sizeof(uint64_t);
	double totalSum = 0;
	bool reduce = true;

	cudaMalloc((void**) &partialSums_d, size);
	partialSums = (uint64_t *) malloc(size);
	// cudaMemcpy(partialSums_d, partialSums, size, cudaMemcpyHostToDevice);

	// call the kernel
	generatePoints<<<ceil(generateThreadCount/256.0), 256>>>(partialSums_d, generateThreadCount, sampleSize);

	if (reduce)
	{
		uint64_t *partialTotals_d;
		uint64_t *partialTotals;
		int sizeTotals = reduceThreadCount * sizeof(uint64_t);
		cudaMalloc((void**) &partialTotals_d, sizeTotals);
		partialTotals = (uint64_t *) malloc(sizeTotals);
		// cudaMemcpy(partialSums_d, partialSums, size, cudaMemcpyHostToDevice);
		reduceCounts<<<ceil(reduceThreadCount/256.0), 256>>>(partialSums_d, partialTotals_d, generateThreadCount, reduceSize);
		cudaMemcpy(partialTotals, partialTotals_d, sizeTotals, cudaMemcpyDeviceToHost);
		for (uint64_t i = 0; i < reduceThreadCount; i++)
		{
			totalSum += partialTotals[i];
			// printf("partialTotals[%i] = %llu\n", i, partialTotals[i]);
		}
	}
	else
	{
		cudaMemcpy(partialSums, partialSums_d, size, cudaMemcpyDeviceToHost);

		// calculate total sum
		for (uint64_t i = 0; i < generateThreadCount; i++)
		{
			totalSum += partialSums[i];
			// printf("partialSums[%i] = %llu\n", i, partialSums[i]);
		}
	}

	// totalSum/(size * # threads) should get us pi/4
	approxPi = ((double)totalSum)/(sampleSize * generateThreadCount);
	approxPi = approxPi * 4.0f;

	cudaFree(partialSums_d);

	return approxPi;
}
