nvcc -O2 --cl-version 2010 --use-local-env -Xcompiler /EHsc,/Zi,/MT -L fftw-3.3.4-dll64 -lcufft -llibfftw3f-3 convolutionFFT2D.cu main.cpp fmvd_deconvolve_common.cpp 