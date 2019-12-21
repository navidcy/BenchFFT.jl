using FFTW, BenchmarkTools, FourierFlows, Random
using LinearAlgebra: mul!, ldiv!
using Random: seed!

seed!(1234) # for reproducibility

Lx, Ly = 2π, 3π
nx, ny = 256, 256
T, tolerance = Float64, 1e-12
# T, tolerance = Float32, 1e-6

gr =  TwoDGrid(nx, Lx, ny, Ly, T=T)

# test function used in bench tests
f = real(ifft(rand(T, (nx, ny)) + im*rand(T, (nx, ny))))
fcomplex  = Complex.(f)

# initialize empty arrays
dfdx = zeros(T, (nx, ny))
dfdx_c = zeros(Complex{T}, (nx, ny))
 fh  = zeros(Complex{T}, (gr.nk, gr.nl))
 fhr = zeros(Complex{T}, (gr.nkr, gr.nl))

# various ways of computing ∂f/∂x

function dx_using_fft(f)
  fh = fft(f)
  dfdx = real(ifft(im*gr.k .* fh))
  return dfdx
end

function dx_using_rfft(f)
  fhr = rfft(f)
  dfdx = irfft(im*gr.kr .* fhr, nx)
  return dfdx
end

effort = FFTW.PATIENT
FFTW.set_num_threads(Sys.CPU_THREADS)
fftplan = plan_fft(Array{T, 2}(undef, nx, ny), flags=effort)
rfftplan = plan_rfft(Array{T, 2}(undef, nx, ny), flags=effort)

function dx_using_fftplan(f)
  fh = fftplan*f
  @. fh = im*gr.k * fh
  dfdx = fftplan \ fh
  return real.(dfdx)
end

function dx_using_rfftplan(f)
  fhr = rfftplan*f
  @. fhr = im*gr.kr * fhr
  dfdx = rfftplan \ fhr
  return dfdx
end

function dx_using_fftplan_mul(f)
  mul!(fh, fftplan, fcomplex) #fftplan within mul! only works if all arrays are complex-valued
  @. fh = im*gr.k * fh
  ldiv!(dfdx_c, fftplan, fh) #fftplan within ldiv! only works if all arrays are complex-valued
  return real.(dfdx_c)
end

function dx_using_rfftplan_mul(f)
  mul!(fhr, rfftplan, f)
  @. fhr = im*gr.kr * fhr
  ldiv!(dfdx, rfftplan, fhr)
  return dfdx
end

dfdx1 = dx_using_fft(f)
dfdx2 = dx_using_rfft(f)
dfdx3 = dx_using_fftplan(f)
dfdx4 = dx_using_rfftplan(f)
dfdx5 = dx_using_fftplan_mul(f)
dfdx6 = dx_using_rfftplan_mul(f)

if (isapprox(dfdx1, dfdx2, rtol=tolerance) && isapprox(dfdx1, dfdx3, rtol=tolerance) 
    && isapprox(dfdx1, dfdx4, rtol=tolerance) && isapprox(dfdx1, dfdx5, rtol=tolerance) 
    && isapprox(dfdx1, dfdx6, rtol=tolerance))
 #make sure that all functions compute ∂f/∂x the same 
  
  println("Performing bench tests for 2D FFTs using nx=", nx, " and ny=", ny, " grid-points with ", T, " arithmetic.")
  println(" ")
  println("computing ∂f/∂x using fft")
  @btime dx_using_fft(f);
  println(" ")
  println("computing ∂f/∂x using rfft")
  @btime dx_using_rfft(f);
  println(" ")
  println("computing ∂f/∂x using fftplan")
  @btime dx_using_fftplan(f);
  println(" ")
  println("computing ∂f/∂x using rfftplan")
  @btime dx_using_rfftplan(f);
  println(" ")
  println("computing ∂f/∂x using fftplan & mul!")
  @btime dx_using_fftplan_mul(f);
  println(" ")
  println("computing ∂f/∂x using rfftplan & mul!")
  @btime dx_using_rfftplan_mul(f);
else
  error("something went wrong while computing the derivatives")
end
