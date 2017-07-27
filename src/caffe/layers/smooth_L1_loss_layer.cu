// ------------------------------------------------------------------
// Fast R-CNN
// copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// Modified by Wei Liu
// ------------------------------------------------------------------

#include <vector>

#include "caffe/layers/smooth_L1_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
__global__ void SmoothL1Forward(const int n, const Ftype* in, Ftype* out) {
  // f(x) = 0.5 * x^2    if |x| < 1
  //        |x| - 0.5    otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Ftype val = in[index];
    Ftype abs_val = abs(val);
    if (abs_val < 1) {
      out[index] = 0.5 * val * val;
    } else {
      out[index] = abs_val - 0.5;
    }
  }
}

template <typename Ftype, typename Btype>
void SmoothL1LossLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub<Ftype>(
      count,
      bottom[0]->gpu_data<Ftype>(),
      bottom[1]->gpu_data<Ftype>(),
      diff_.template mutable_gpu_data<Ftype>());    // d := b0 - b1
  if (has_weights_) {
    caffe_gpu_mul(
        count,
        bottom[2]->gpu_data<Ftype>(),
        diff_.template gpu_data<Ftype>(),
        diff_.template mutable_gpu_data<Ftype>());  // d := w * (b0 - b1)
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  SmoothL1Forward<Ftype, Btype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.template gpu_data<Ftype>(), errors_.template mutable_gpu_data<Ftype>());
  CUDA_POST_KERNEL_CHECK;

  Ftype loss;
  caffe_gpu_asum(count, errors_.template gpu_data<Ftype>(), &loss);
  top[0]->mutable_cpu_data<Ftype>()[0] = loss / bottom[0]->num();
}

template <typename Ftype, typename Btype>
__global__ void SmoothL1Backward(const int n, const Btype* in, Btype* out) {
  // f'(x) = x         if |x| < 1
  //       = sign(x)   otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Btype val = in[index];
    Btype abs_val = abs(val);
    if (abs_val < 1) {
      out[index] = val;
    } else {
      out[index] = (Btype(0) < val) - (val < Btype(0));
    }
  }
}

template <typename Ftype, typename Btype>
void SmoothL1LossLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  int count = diff_.count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SmoothL1Backward<Ftype, Btype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.template gpu_data<Btype>(), diff_.template mutable_gpu_data<Btype>());
  CUDA_POST_KERNEL_CHECK;
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Btype sign = (i == 0) ? 1 : -1;
      const Btype alpha = sign * top[0]->cpu_diff<Btype>()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                           // alpha
          diff_.template gpu_data<Btype>(),                // x
          Btype(0),                        // beta
          bottom[i]->mutable_gpu_diff<Btype>());  // y
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(SmoothL1LossLayer);

}  // namespace caffe
