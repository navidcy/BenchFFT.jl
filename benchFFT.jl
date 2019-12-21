using FFTW, BenchmarkTools, FourierFlows, Random
using LinearAlgebra: mul!, ldiv!
using Random: seed!

seed!(1234) # for reproducibility

Lx, Ly = 2π, 3π
nx, ny = 128, 256

gr =  TwoDGrid(nx, Lx, ny, Ly)

fh_test = rand(nx, ny) + im*rand(nx, ny)

f = real(ifft(fh_test))

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
fftplan = plan_fft(Array{Complex{Float64}, 2}(undef, nx, ny), flags=effort)
rfftplan = plan_rfft(Array{Float64, 2}(undef, nx, ny), flags=effort)

function dx_using_fftplan(f)
  mul!(fh, fftplan, Complex.(f))
  @. fh = im*gr.k * fh
  ldiv!(dfdx_c, fftplan, fh)
  return real.(dfdx_c)
end

function dx_using_rfftplan(f)
  mul!(fhr, rfftplan, f)
  @. fhr = im*gr.kr * fhr
  ldiv!(dfdx, rfftplan, fhr)
  return dfdx
end

dxf1 = dx_using_fft(f)
dxf2 = dx_using_rfft(f)
dxf3 = dx_using_fftplan(f)
dxf4 = dx_using_rfftplan(f)

if  isapprox(dxf1, dxf2, rtol=1e-12) && isapprox(dxf1, dxf3, rtol=1e-12) && isapprox(dxf1, dxf4, rtol=1e-12)
  println("compute derivatives using fft")
  @btime dx_using_fft(f);
  println(" ")
  println("compute derivatives using rfft")
  @btime dx_using_rfft(f);
  println(" ")
  println("compute derivatives using fftplan")
  @btime dx_using_fftplan(f);
  println(" ")
  println("compute derivatives using rfftplan")
  @btime dx_using_rfftplan(f);
else
  error("something's wrong with computing the derivatives")
end
  
  

