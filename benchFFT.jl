using FFTW, BenchmarkTools, FourierFlows, Random
using LinearAlgebra: mul!, ldiv!
using Random: seed!

seed!(1234) # for reproducibility

Lx, Ly = 2π, 3π
nx, ny = 512, 512

gr =  TwoDGrid(nx, Lx, ny, Ly)

# test function used in bench tests
f = real(ifft(rand(nx, ny) + im*rand(nx, ny)))
fcomplex  = Complex.(f)

# initialize empty arrays
dfdx = zeros(nx, ny)
dfdx_c = zeros(Complex{Float64}, (nx, ny))
 fh  = zeros(Complex{Float64}, (gr.nk, gr.nl))
 fhr = zeros(Complex{Float64}, (gr.nkr, gr.nl))

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
fftplan = plan_fft(Array{Float64, 2}(undef, nx, ny), flags=effort)
rfftplan = plan_rfft(Array{Float64, 2}(undef, nx, ny), flags=effort)

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
  mul!(fh, fftplan, fcomplex) #fftplan within mul! only works for complex-valued arreys
  @. fh = im*gr.k * fh
  ldiv!(dfdx_c, fftplan, fh)
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

if  isapprox(dfdx1, dfdx2, rtol=1e-13) && isapprox(dfdx1, dfdx3, rtol=1e-13) && isapprox(dfdx1, dfdx4, rtol=1e-13) && isapprox(dfdx1, dfdx5, rtol=1e-13) && isapprox(dfdx1, dfdx6, rtol=1e-13) #make sure that all functions compute ∂f/∂x the same 
  
  println("Performing bench tests with nx=", nx, " and ny=", ny, " grid-points")
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
  println("computing ∂f/∂x using fftplan + mul!")
  @btime dx_using_fftplan_mul(f);
  println(" ")
  println("computing ∂f/∂x using rfftplan + mul!")
  @btime dx_using_rfftplan_mul(f);
else
  error("something went wrong with computing the derivatives")
end
  
  

