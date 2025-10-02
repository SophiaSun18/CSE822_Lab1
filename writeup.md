# CSE 822 Lab 1 Write-up

## Compile with the following command on `unicorn`:
g++ mandelbrot_cpu.cpp -o cpu_output -mavx2
nvcc mandelbrot_gpu.cu -o gpu_output -arch=sm_89

## Question 1
How does the run time of the scalar GPU implementation compare to the scalar CPU implementation? What factors do you think might contribute to each of their run times?

GPU scalar: 291.039ms
CPU scalar: 99.2118ms
GPU scalar runs significantly slower than CPU scalar by almost 3 times. My idea is that running on GPU requires extra overhead to allocate blocks and threads, launch kernels and transfer memory between CPU and GPU. However in this program we only use 1 block and 1 thread, which doesn't utilize GPU's capability for highly parallel execution.

## Question 2
How did you initialize `cx` and `cy`? How did you handle differences in the number of inner loop iterations between pixels in the same vector? How does the performance of your vectorized implementation compare to the scalar versions?

I implemented `cx` and `cy` using _m256 vectors (as 512 is not supported on this machine), collecting 8 pixels of i and j respectively and store them into `cx` and `cy` after some arithmetic operations.
I used an active mask to keep track of pixels that are done with computation (inactive) and pixels that still need further computation (active). In each iteration the active mask is updated until the max # of iteration is reached or all the pixels become inactive (done).

Testing with image size 256x256 and 1000 max iterations.
Running mandelbrot_cpu_scalar ...
  Runtime: 82.7809 ms
Running mandelbrot_cpu_vector ...
  Runtime: 35.1548 ms
  Correctness: average output difference from reference = 0

The performance of the vecotrized code is over 50% faster than the scalar code.

## Question 3
How does the performance of your vectorized implementation compare to the scalar versions? Given how you implemented the vector-parallel CPU version with explicit SIMD, how do you think the GPU executes multiple kernel instances that run different numbers of iterations?

Testing with image size 256x256 and 1000 max iterations.
Running launch_mandelbrot_gpu_scalar ...
  Runtime: 291.039 ms
  Correctness: average output difference from reference = 0
Running launch_mandelbrot_gpu_vector ...
  Runtime: 13.8415 ms
  Correctness: average output difference from reference 0

The GPU vectorized implementation is over 20x faster than GPU scalar implementation, and also about 2.7x faster than CPU vectorized implementation.
In the CPU vectorization implementation, different number of iterations on the same vector is handled by using a mask to "mask out" pixels that don't need to be processed in the current iteration. It is possible that GPU uses a similiar technique, but it's masking out threads according to the thread id instead of applying a mask on the vector.
