## Question 1
How does the run time of the scalar GPU implementation compare to the scalar CPU implementation? What factors do you think might contribute to each of their run times?

GPU scalar: 0.000148ms
CPU scalar: 99.2118ms
GPU scalar runs significantly faster than CPU scalar even when running on 1 core and 1 thread, which shows that GPU is probably faster on both computation and memory transfer compared to CPU.

## Question 2
How did you initialize `cx` and `cy`? How did you handle differences in the number of inner loop iterations between pixels in the same vector? How does the performance of your vectorized implementation compare to the scalar versions?

I implemented `cx` and `cy` using _m256 vectors (as 512 is not supported on this machine), collecting 8 pixels of i and j respectively and store them into `cx` and `cy` after some arithmetic operations.
I used an active mask to keep track of pixels that are done with computation (inactive) and pixels that still need further computation (active). In each iteration the active mask is updated until the max # of iteration is reached or all the pixels become inactive (done).

Testing with image size 256x256 and 1000 max iterations.
Running mandelbrot_cpu_scalar ...
  Runtime: 82.8258 ms
Running mandelbrot_cpu_vector ...
  Runtime: 36.6271 ms
  Correctness: average output difference from reference = 0.000982605

The performance of the vecotrized code is over 50% faster than the scalar code, with a slightly output difference probably due to floating point precision issue.

## Question 3
How does the performance of your vectorized implementation compare to the scalar versions? Given how you implemented the vector-parallel CPU version with explicit SIMD, how do you think the GPU executes multiple kernel instances that run different numbers of iterations?

Testing with image size 256x256 and 1000 max iterations.
Running launch_mandelbrot_gpu_scalar ...
  Runtime: 0.00012 ms
  Correctness: average output difference from reference = 0.248333
Running launch_mandelbrot_gpu_vector ...
  Runtime: 0.000125 ms
  Correctness: average output difference from reference 0.248333

Somehow the performance doesn't differ very much. My guess is that the matrix size is too small and for such a small task parallelizing the task doesn't benefit much on GPU.
In the CPU vectorization code, different number of iterations on the same vector is handled by using a mask to "mask out" pixels that don't need to be processed in the current iteration. It is possible that GPU uses a similiar technique, but it's masking out threads according to the thread id instead of applying a mask on the vector.
