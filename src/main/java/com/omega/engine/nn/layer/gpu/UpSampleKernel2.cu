#define BLOCK 1024 

__device__ int translate_idx(int ii, int d1, int d2, int d3, int scale_factor) {
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  w = w/scale_factor;
  z = z/scale_factor;
  d2 /= scale_factor;
  d3 /= scale_factor;
  return (((x*d1+y)*d2)+z)*d3+w;
}

__device__ int translate_idx_inv(
    int ii, int d1, int d2, int d3, int scale_factor, int off_x, int off_y) {
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  w = w*scale_factor+off_x;
  z = z*scale_factor+off_y;
  d2 *= scale_factor;
  d3 *= scale_factor;
  return (((x*d1+y)*d2)+z)*d3+w;
}

extern "C"
__global__ void upscale(const float *input, float *output, int no_elements,
                        int scale_factor, int d1, int d2, int d3) {
  int ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  int ipidx = translate_idx(ii, d1, d2, d3, scale_factor);
  output[ii]=input[ipidx];
}

extern "C"
__global__ void downscale(float *gradInput_data, const float *gradOutput_data,
                          int no_elements, int scale_factor, int d1, int d2, int d3) {
  int ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  for (int i=0; i < scale_factor; i++){
    for(int j=0; j < scale_factor; j++){
      int ipidx = translate_idx_inv(ii, d1, d2, d3, scale_factor, i, j);
      gradInput_data[ii] += gradOutput_data[ipidx];
    }
  }
}

__device__ int translate_idx_3d(int ii, int d1, int d2, int d3, int d4, int scale_factor) {
  int n, c, d, h, w;
  w = ii % d4;
  ii = ii/d4;
  h = ii % d3;
  ii = ii/d3;
  d = ii % d2;
  ii = ii/d2;
  c = ii % d1;
  ii = ii/d1;
  n = ii;
  d = d/scale_factor;
  h = h/scale_factor;
  w = w/scale_factor;

  d2 /= scale_factor;
  d3 /= scale_factor;
  d4 /= scale_factor;

  return (((n*d1+c)*d2+d)*d3+h)*d4+w;
}

extern "C"
__global__ void upscale3d(const float *input, float *output, int no_elements,
                        int scale_factor, int d1, int d2, int d3, int d4) {
  int ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  int ipidx = translate_idx_3d(ii, d1, d2, d3, d4, scale_factor);
  output[ii]=input[ipidx];
}

__device__ int translate_idx_3d_inv(
    int ii, int d1, int d2, int d3, int d4, int scale_factor, int off_d, int off_x, int off_y) {
  int n, c, d, h, w;
  w = ii % d4;
  ii = ii/d4;
  h = ii % d3;
  ii = ii/d3;
  d = ii % d2;
  ii = ii/d2;
  c = ii % d1;
  ii = ii/d1;
  n = ii;
  d = d*scale_factor+off_d;
  w = w*scale_factor+off_x;
  h = h*scale_factor+off_y;
  d2 *= scale_factor;
  d3 *= scale_factor;
  d4 *= scale_factor;
  return (((n*d1+c)*d2+d)*d3+h)*d4+w;
}

extern "C"
__global__ void downscale3d(float *gradInput_data, const float *gradOutput_data,
                          int no_elements, int scale_factor, int d1, int d2, int d3, int d4) {
  int ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  for (int d=0; d < scale_factor; d++){
	  for (int i=0; i < scale_factor; i++){
	    for(int j=0; j < scale_factor; j++){
	      int ipidx = translate_idx_3d_inv(ii, d1, d2, d3, d4, scale_factor, d, i, j);
	      gradInput_data[ii] += gradOutput_data[ipidx];
	    }
	  }
  }
}