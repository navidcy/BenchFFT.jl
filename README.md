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
```julia
Performing bench tests with nx=512 and ny=512 grid-points...

computing ∂f/∂x using fft
  6.522 ms (136 allocations: 18.02 MiB)

computing ∂f/∂x using rfft
  3.957 ms (155 allocations: 10.04 MiB)

computing ∂f/∂x using fftplan
  4.685 ms (19 allocations: 14.00 MiB)

computing ∂f/∂x using rfftplan
  1.525 ms (11 allocations: 6.02 MiB)

computing ∂f/∂x using fftplan + mul!
  3.557 ms (13 allocations: 2.00 MiB)

computing ∂f/∂x using rfftplan + mul!
  887.064 μs (5 allocations: 224 bytes)
```
