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
void SmoothL1LossLayer<Ftype, Btype>::LayerSetUp(
  const vector<Blob*>& bottom, const vector<Blob*>& top) {
  LossLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  has_weights_ = (bottom.size() == 3);
}

template <typename Ftype, typename Btype>
void SmoothL1LossLayer<Ftype, Btype>::Reshape(
  const vector<Blob*>& bottom, const vector<Blob*>& top) {
  LossLayer<Ftype, Btype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  if (has_weights_) {
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
  }
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  errors_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Ftype, typename Btype>
void SmoothL1LossLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data<Ftype>(),
      bottom[1]->cpu_data<Ftype>(),
      diff_.template mutable_cpu_data());
  if (has_weights_) {
    caffe_mul(
        count,
        bottom[2]->cpu_data<Ftype>(),
        diff_.template cpu_data<Ftype>(),
        diff_.template mutable_cpu_data<Ftype>());  // d := w * (b0 - b1)
  }
  const Ftype* diff_data = diff_.template cpu_data<Ftype>();
  Ftype* error_data = errors_.template mutable_cpu_data<Ftype>();
  for (int i = 0; i < count; ++i) {
    Ftype val = diff_data[i];
    Ftype abs_val = fabs(val);
    if (abs_val < 1.) {
      error_data[i] = 0.5 * val * val;
    } else {
      error_data[i] = abs_val - 0.5;
    }
  }
  top[0]->mutable_cpu_data<Ftype>()[0] =
      caffe_cpu_asum(count, errors_.template cpu_data()) / bottom[0]->num();
}

template <typename Ftype, typename Btype>
void SmoothL1LossLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  int count = diff_.count();
  Btype* diff_data = diff_.template mutable_cpu_data<Btype>();
  for (int i = 0; i < count; ++i) {
    Btype val = diff_data[i];
    // f'(x) = x         if |x| < 1
    //       = sign(x)   otherwise
    if (fabs(val) < 1.) {
      diff_data[i] = val;
    } else {
      diff_data[i] = (Btype(0) < val) - (val < Btype(0));
    }
  }
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Btype sign = (i == 0) ? 1 : -1;
      const Btype alpha = sign * top[0]->cpu_diff<Btype>()[0] / bottom[i]->num();
      caffe_cpu_axpby<Btype>(
          bottom[i]->count(),               // count
          alpha,                            // alpha
          diff_.template cpu_data<Btype>(),                 // a
          Btype(0),                         // beta
          bottom[i]->mutable_cpu_diff<Btype>());   // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1LossLayer);
#endif

INSTANTIATE_CLASS_FB(SmoothL1LossLayer);
REGISTER_LAYER_CLASS(SmoothL1Loss);

}  // namespace caffe
