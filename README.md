# BenchFFT.jl
A project comparing how different FFT implementations perform


Run
```julia
julia --project -e 'import Pkg; Pkg.instantiate();'
```
to install all dependencies and then
```julia
julia --project benchFFT.jl
```
to run the benchmark.


On a 2.9 GHz Intel Core i7 Quad-Core, 16MB RAM MacbookPro one gets:
```
Performing bench tests for 2D FFTs using nx=256 and ny=256 grid-points with Float64 arithmetic.

computing ∂f/∂x using fft
  1.785 ms (130 allocations: 4.51 MiB)

computing ∂f/∂x using rfft
  1.677 ms (147 allocations: 2.53 MiB)

computing ∂f/∂x using fftplan
  742.876 μs (19 allocations: 3.50 MiB)

computing ∂f/∂x using rfftplan
  313.398 μs (11 allocations: 1.51 MiB)

computing ∂f/∂x using fftplan & mul!
  423.025 μs (13 allocations: 512.45 KiB)

computing ∂f/∂x using rfftplan & mul!
  229.309 μs (5 allocations: 224 bytes)
```
and
```
Performing bench tests for 2D FFTs using nx=256 and ny=256 grid-points with Float32 arithmetic.

computing ∂f/∂x using fft
  1.569 ms (130 allocations: 2.26 MiB)

computing ∂f/∂x using rfft
  1.579 ms (147 allocations: 1.27 MiB)

computing ∂f/∂x using fftplan
  503.819 μs (19 allocations: 1.75 MiB)

computing ∂f/∂x using rfftplan
  233.894 μs (11 allocations: 772.45 KiB)

computing ∂f/∂x using fftplan & mul!
  340.899 μs (13 allocations: 256.45 KiB)

computing ∂f/∂x using rfftplan & mul!
  196.207 μs (5 allocations: 224 bytes)
```
