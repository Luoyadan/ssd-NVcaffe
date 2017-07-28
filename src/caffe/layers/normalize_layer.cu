#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/filler.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// divid a matrix with vector
template <typename Dtype>
__global__ void DivBsx(const int nthreads, const Dtype* A,
    const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
    Dtype* B) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = index % cols;
    int r = (index / cols) % rows;
    if (trans == CblasNoTrans) {
      B[index] = A[index] / v[c];
    } else {
      B[index] = A[index] / v[r];
    }
  }
}

template <typename Dtype>
__global__ void MulBsx(const int nthreads, const Dtype* A,
    const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
    Dtype* B) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = index % cols;
    int r = (index / cols) % rows;
    if (trans == CblasNoTrans) {
      B[index] = A[index] * v[c];
    } else {
      B[index] = A[index] * v[r];
    }
  }
}

template <typename Ftype, typename Btype>
void NormalizeLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->template gpu_data<Ftype>();
  Ftype* top_data = top[0]->template mutable_gpu_data<Ftype>();
  Ftype* buffer_data = buffer_.template mutable_gpu_data<Ftype>();
  Ftype* norm_data;
  if (across_spatial_) {
    // need to index it
    norm_data = norm_.template mutable_cpu_data<Ftype>();
  } else {
    norm_data = norm_.template mutable_gpu_data<Ftype>();
    // add eps to avoid overflow
    caffe_gpu_set<Ftype>(norm_.count(), Ftype(eps_), norm_data);
  }
  const Ftype* scale;
  if (channel_shared_) {
    scale = this->blobs_[0]->template cpu_data<Ftype>();
  } else {
    scale = this->blobs_[0]->template gpu_data<Ftype>();
  }
  const Ftype* sum_channel_multiplier = sum_channel_multiplier_.template gpu_data<Ftype>();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  int channels = bottom[0]->channels();
  for (int n = 0; n < num; ++n) {
    caffe_gpu_powx<Ftype>(dim, bottom_data, Ftype(2), buffer_data);
    if (across_spatial_) {
      Ftype normsqr;
      caffe_gpu_asum<Ftype>(dim, buffer_data, &normsqr);
      // add eps to avoid overflow
      norm_data[n] = pow(static_cast<float>(normsqr+eps_), 0.5);
      caffe_gpu_scale<Ftype>(dim, Ftype(1.0 / norm_data[n]), bottom_data,
                             top_data);
    } else {
      // compute norm
      caffe_gpu_gemv<Ftype>(CblasTrans, channels, spatial_dim, Ftype(1),
                            buffer_data, sum_channel_multiplier, Ftype(1),
                            norm_data);
      caffe_gpu_powx<Ftype>(spatial_dim, norm_data, Ftype(0.5), norm_data);
      // scale the layer
      // NOLINT_NEXT_LINE(whitespace/operators)
      DivBsx<Ftype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
          dim, bottom_data, norm_data, channels, spatial_dim, CblasNoTrans,
          top_data);
      CUDA_POST_KERNEL_CHECK;
      norm_data += spatial_dim;
    }
    // scale the output
    if (channel_shared_) {
      caffe_gpu_scal<Ftype>(dim, scale[0], top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      MulBsx<Ftype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
          dim, top_data, scale, channels, spatial_dim, CblasTrans,
          top_data);
      CUDA_POST_KERNEL_CHECK;
    }
    bottom_data += dim;
    top_data += dim;
  }
}

template <typename Ftype, typename Btype>
void NormalizeLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  const Btype* top_diff = top[0]->template gpu_diff<Btype>();
  const Btype* top_data = top[0]->template gpu_data<Btype>();
  const Btype* bottom_data = bottom[0]->template mutable_gpu_data<Btype>();
  Btype* bottom_diff = bottom[0]->template mutable_gpu_diff<Btype>();
  const Btype* norm_data;
  if (across_spatial_) {
    // need to index it
    norm_data = norm_.template cpu_data<Btype>();
  } else {
    norm_data = norm_.template gpu_data<Btype>();
  }
  const Btype* scale;
  if (channel_shared_) {
    scale = this->blobs_[0]->template cpu_data<Btype>();
  } else {
    scale = this->blobs_[0]->template gpu_data<Btype>();
  }
  Btype* buffer_data = buffer_.template mutable_gpu_data<Btype>();
  Btype* buffer_channel = buffer_channel_.template mutable_gpu_data<Btype>();
  Btype* buffer_spatial = buffer_spatial_.template mutable_gpu_data<Btype>();
  const Btype* sum_channel_multiplier = sum_channel_multiplier_.template gpu_data<Btype>();
  const Btype* sum_spatial_multiplier = sum_spatial_multiplier_.template gpu_data<Btype>();
  int count = top[0]->count();
  int num = top[0]->num();
  int dim = count / num;
  int spatial_dim = top[0]->height() * top[0]->width();
  int channels = top[0]->channels();

  // Propagate to param
  if (this->param_propagate_down_[0]) {
    if (channel_shared_) {
      Btype* scale_diff = this->blobs_[0]->template mutable_cpu_diff<Btype>();
      Btype a;
      caffe_gpu_dot<Btype>(count, top_data, top_diff, &a);
      scale_diff[0] += a / scale[0];
    } else {
      Btype* scale_diff = this->blobs_[0]->template mutable_gpu_diff<Btype>();
      for (int n = 0; n < num; ++n) {
        // compute a
        caffe_gpu_mul<Btype>(dim, top_data+n*dim, top_diff+n*dim, buffer_data);
        caffe_gpu_gemv<Btype>(CblasNoTrans, channels, spatial_dim, Btype(1),
                              buffer_data, sum_spatial_multiplier, Btype(0),
                              buffer_channel);
        // store a / scale[i] in buffer_data temporary
        caffe_gpu_div<Btype>(channels, buffer_channel, scale, buffer_channel);
        caffe_gpu_add<Btype>(channels, buffer_channel, scale_diff, scale_diff);
      }
    }
  }

  // Propagate to bottom
  if (propagate_down[0]) {
    for (int n = 0; n < num; ++n) {
      if (across_spatial_) {
        Btype a;
        caffe_gpu_dot<Btype>(dim, bottom_data, top_diff, &a);
        caffe_gpu_scale<Btype>(dim, a / norm_data[n] / norm_data[n],
                               bottom_data, bottom_diff);
        caffe_gpu_sub<Btype>(dim, top_diff, bottom_diff, bottom_diff);
        caffe_gpu_scale<Btype>(dim, Btype(1.0 / norm_data[n]), bottom_diff,
                               bottom_diff);
      } else {
        // dot product between bottom_data and top_diff
        caffe_gpu_mul<Btype>(dim, bottom_data, top_diff, buffer_data);
        caffe_gpu_gemv<Btype>(CblasTrans, channels, spatial_dim, Btype(1),
                              buffer_data, sum_channel_multiplier, Btype(0),
                              buffer_spatial);
        // scale botom_diff
        // NOLINT_NEXT_LINE(whitespace/operators)
        MulBsx<Btype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
            dim, bottom_data, buffer_spatial, channels, spatial_dim,
            CblasNoTrans, bottom_diff);
        CUDA_POST_KERNEL_CHECK;
        // divide by square of norm
        caffe_gpu_powx<Btype>(spatial_dim, norm_data, Btype(2), buffer_spatial);
        // NOLINT_NEXT_LINE(whitespace/operators)
        DivBsx<Btype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
            dim, bottom_diff, buffer_spatial, channels, spatial_dim,
            CblasNoTrans, bottom_diff);
        CUDA_POST_KERNEL_CHECK;
        caffe_gpu_sub<Btype>(dim, top_diff, bottom_diff, bottom_diff);
        // divide by norm
        // NOLINT_NEXT_LINE(whitespace/operators)
        DivBsx<Btype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
            dim, bottom_diff, norm_data, channels, spatial_dim, CblasNoTrans,
            bottom_diff);
        CUDA_POST_KERNEL_CHECK;
        norm_data += spatial_dim;
      }
      // scale the diff
      if (channel_shared_) {
        caffe_gpu_scal<Btype>(dim, scale[0], bottom_diff);
      } else {
        // NOLINT_NEXT_LINE(whitespace/operators)
        MulBsx<Btype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
            dim, bottom_diff, scale, channels, spatial_dim, CblasTrans,
            bottom_diff);
        CUDA_POST_KERNEL_CHECK;
      }
      bottom_data += dim;
      top_diff += dim;
      bottom_diff += dim;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(NormalizeLayer);


}  // namespace caffe
