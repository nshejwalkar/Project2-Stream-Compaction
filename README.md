**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 2 - Stream Compaction**

* Neel Shejwalkar
  * [LinkedIn](https://www.linkedin.com/in/neel-shejwalkar/), [twitter](https://x.com/neelshej)
* Tested on: Windows 11, i7-10510U @ 1.80GHz 16GB, MX250 (GP108, sm_61) 2GB (Personal Computer)

| | |
|---|---|
| SMs | 3 |
| CUDA cores | 128/SM = 384 total |
| Max threads / SM | 2048 (64 warps) |
| Max blocks / SM | 32 |
| L2 cache | 512 KB |
| Shared memory | 96 KB/SM, 48 KB/block |
| Registers | 65,536 per SM |
| Memory bus | 64-bit @ 3004 MHz = 48.1 GB/s |
| VRAM | 2048 MB |
| SM clock (max) | 1582 MHz |
| Peak FP32 | 1215 GFLOP/s |

# Performance Analysis

![Runtime vs. block size](img/perf_blocksize.png)

![Runtime vs. array size](img/perf_size.png)

# Test Output

```
****************
** SCAN TESTS **
****************
    [   0   7  19  28   4  30  16  37  46  38  18   6  42 ...   9   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0009ms    (std::chrono Measured)
    [   0   0   7  26  54  58  88 104 141 187 225 243 249 ... 6607 6616 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0007ms    (std::chrono Measured)
    [   0   0   7  26  54  58  88 104 141 187 225 243 249 ... 6532 6568 ]
    passed 
==== naive scan, power-of-two ====
   elapsed time: 0.31744ms    (CUDA Measured)
    passed 
==== naive scan, non-power-of-two ====
   elapsed time: 0.1536ms    (CUDA Measured)
    passed 
==== work-efficient scan, power-of-two ====
   elapsed time: 0.411648ms    (CUDA Measured)
    passed 
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.345088ms    (CUDA Measured)
    passed 
==== thrust scan, power-of-two ====
   elapsed time: 0.245824ms    (CUDA Measured)
    passed 
==== thrust scan, non-power-of-two ====
   elapsed time: 0.203776ms    (CUDA Measured)
    passed 

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   1   0   0   2   2   3   2   0   2   0   2 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0013ms    (std::chrono Measured)
    [   2   3   1   2   2   3   2   2   2   3   3   1   2 ...   1   3 ]
    passed 
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.001ms    (std::chrono Measured)
    [   2   3   1   2   2   3   2   2   2   3   3   1   2 ...   2   1 ]
    passed 
==== cpu compact with scan ====
   elapsed time: 0.0021ms    (std::chrono Measured)
    [   2   3   1   2   2   3   2   2   2   3   3   1   2 ...   1   3 ]
    passed 
==== work-efficient compact, power-of-two ====
   elapsed time: 0.652288ms    (CUDA Measured)
    passed 
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.348768ms    (CUDA Measured)
    passed 
```
