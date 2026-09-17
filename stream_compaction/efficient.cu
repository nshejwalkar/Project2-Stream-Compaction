#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define EFFICIENT_BLOCK_SIZE 512

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // only launch one thread per active node, so idx gets stretched back out to the array index k
        __global__ void kernUpSweep(int active, int stride, int *data) {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx >= active) return;

            int k = idx * stride * 2;
            data[k + stride * 2 - 1] += data[k + stride - 1];  // absorb element from last iteration
        }

        __global__ void kernDownSweep(int active, int stride, int *data) {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx >= active) return;

            int k = idx * stride * 2;
            int t = data[k + stride - 1];
            data[k + stride - 1] = data[k + stride * 2 - 1];  // swap
            data[k + stride * 2 - 1] += t;  // absorb
        }

        // in place exclusive scan = upsweep then downsweep, on an array already padded to a power of 2
        void scanDevice(int n_padded, int *dev_idata) {
            int levels = ilog2(n_padded);

            for (int d = 0; d < levels; d++) {
                int active = n_padded >> (d + 1);
                dim3 fullBlocksPerGrid((active + EFFICIENT_BLOCK_SIZE - 1) / EFFICIENT_BLOCK_SIZE);
                kernUpSweep<<<fullBlocksPerGrid, EFFICIENT_BLOCK_SIZE>>>(active, 1 << d, dev_idata);
            }

            // set the last element to 0 before downsweep
            cudaMemset(dev_idata + n_padded - 1, 0, sizeof(int));

            for (int d = levels - 1; d >= 0; d--) {
                int active = n_padded >> (d + 1);
                dim3 fullBlocksPerGrid((active + EFFICIENT_BLOCK_SIZE - 1) / EFFICIENT_BLOCK_SIZE);
                kernDownSweep<<<fullBlocksPerGrid, EFFICIENT_BLOCK_SIZE>>>(active, 1 << d, dev_idata);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int n_padded = 1 << ilog2ceil(n);

            int* dev_idata;
            cudaMalloc((void**)&dev_idata, n_padded * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemset(dev_idata, 0, n_padded * sizeof(int));  // automatically pads
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            scanDevice(n_padded, dev_idata);  // no need for the right shift like with the naive version
            timer().endGpuTimer();
            checkCUDAError("work-efficient scan failed!");

            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int n_padded = 1 << ilog2ceil(n);
            
            // input array
            int* dev_idata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            // output array
            int* dev_odata;
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            
            // boolean array
            int* dev_bools;
            cudaMalloc((void**)&dev_bools, n_padded * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            cudaMemset(dev_bools, 0, n_padded * sizeof(int));
    
            // index array
            int* dev_indices;
            cudaMalloc((void**)&dev_indices, n_padded * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");

            dim3 fullBlocksPerGrid((n + EFFICIENT_BLOCK_SIZE - 1) / EFFICIENT_BLOCK_SIZE);

            timer().startGpuTimer();
            Common::kernMapToBoolean<<<fullBlocksPerGrid, EFFICIENT_BLOCK_SIZE>>>(n, dev_bools, dev_idata);
            cudaMemcpy(dev_indices, dev_bools, n_padded * sizeof(int), cudaMemcpyDeviceToDevice);
            scanDevice(n_padded, dev_indices);
            Common::kernScatter<<<fullBlocksPerGrid, EFFICIENT_BLOCK_SIZE>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            timer().endGpuTimer();
            checkCUDAError("work-efficient compact failed!");

            // just like the cpu version, we need to get the last bool and index to compute the count
            int lastBool, lastIndex;
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastBool + lastIndex;

            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            return count;
        }
    }
}
