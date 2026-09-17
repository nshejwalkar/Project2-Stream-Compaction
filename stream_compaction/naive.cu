#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define NAIVE_BLOCK_SIZE 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernShiftRight(int n, int *odata, const int *idata) {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx >= n) return;

            odata[idx] = idx == 0 ? 0 : idata[idx - 1];
        }

        __global__ void kernScanPass(int n, int offset, int *odata, const int *idata) {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx >= n) return;

            odata[idx] = idx >= offset ? idata[idx - offset] + idata[idx] : idata[idx];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_A;
            int* dev_B;
            cudaMalloc((void**)&dev_A, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_A failed!");
            cudaMalloc((void**)&dev_B, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_B failed!");
            cudaMemcpy(dev_B, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 fullBlocksPerGrid((n + NAIVE_BLOCK_SIZE - 1) / NAIVE_BLOCK_SIZE);

            timer().startGpuTimer();
            // shift first so the inclusive passes below end up producing an exclusive scan
            kernShiftRight<<<fullBlocksPerGrid, NAIVE_BLOCK_SIZE>>>(n, dev_A, dev_B);
            for (int offset = 1; offset < n; offset <<= 1) {  // log_2 n passes
                kernScanPass<<<fullBlocksPerGrid, NAIVE_BLOCK_SIZE>>>(n, offset, dev_B, dev_A);
                std::swap(dev_A, dev_B);
            }
            timer().endGpuTimer(); 
            checkCUDAError("naive scan failed!");

            cudaMemcpy(odata, dev_A, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_A);
            cudaFree(dev_B);
        }
    }
}
