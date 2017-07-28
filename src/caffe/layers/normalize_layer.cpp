#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/normalize_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void NormalizeLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  buffer_.Reshape(1, bottom[0]->channels(),
                   bottom[0]->height(), bottom[0]->width());
  buffer_channel_.Reshape(1, bottom[0]->channels(), 1, 1);
  buffer_spatial_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  NormalizeParameter norm_param = this->layer_param().norm_param();
  across_spatial_ = norm_param.across_spatial();
  if (across_spatial_) {
    norm_.Reshape(bottom[0]->num(), 1, 1, 1);
  } else {
    norm_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  }
  eps_ = norm_param.eps();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->width() * bottom[0]->height();
  sum_channel_multiplier_.Reshape(1, channels, 1, 1);
  caffe_set<Ftype>(channels, Ftype(1), sum_channel_multiplier_.template mutable_cpu_data<Ftype>());
  sum_spatial_multiplier_.Reshape(
      1, 1, bottom[0]->height(), bottom[0]->width());
  caffe_set<Ftype>(spatial_dim, Ftype(1), sum_spatial_multiplier_.template mutable_cpu_data<Ftype>());
  channel_shared_ = norm_param.channel_shared();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    if (channel_shared_) {
      this->blobs_[0] = Blob::create<Ftype>(vector<int>(0));
	  
    } else {
      this->blobs_[0] = Blob::create<Ftype>(vector<int>(1, channels));
	  
    }
    shared_ptr<Filler<Ftype> > scale_filler;
    if (norm_param.has_scale_filler()) {
      scale_filler.reset(GetFiller<Ftype>(norm_param.scale_filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(1.0);
      scale_filler.reset(GetFiller<Ftype>(filler_param));
    }
    scale_filler->Fill(this->blobs_[0].get());
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "Scale size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "Scale size is inconsistent with prototxt config";
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Ftype, typename Btype>
void NormalizeLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  buffer_.Reshape(1, bottom[0]->channels(),
                   bottom[0]->height(), bottom[0]->width());
  if (!across_spatial_) {
    norm_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  }
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  if (spatial_dim != sum_spatial_multiplier_.count()) {
    sum_spatial_multiplier_.Reshape(
        1, 1, bottom[0]->height(), bottom[0]->width());
    caffe_set<Ftype>(spatial_dim, Ftype(1),
              sum_spatial_multiplier_.template mutable_cpu_data<Ftype>());
    buffer_spatial_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  }
}

template <typename Ftype, typename Btype>
void NormalizeLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->template cpu_data<Ftype>();
  Ftype* top_data = top[0]->template mutable_cpu_data<Ftype>();
  const Ftype* scale = this->blobs_[0]->template cpu_data<Ftype>();
  Ftype* buffer_data = buffer_.template mutable_cpu_data<Ftype>();
  Ftype* norm_data = norm_.template mutable_cpu_data<Ftype>();
  // add eps to avoid overflow
  caffe_set<Ftype>(norm_.count(), float(eps_), norm_data);
  const Ftype* sum_channel_multiplier = sum_channel_multiplier_.template cpu_data();
  const Ftype* sum_spatial_multiplier = sum_spatial_multiplier_.template cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  int channels = bottom[0]->channels();
  for (int n = 0; n < num; ++n) {
    caffe_sqr<Ftype>(dim, bottom_data, buffer_data);
    if (across_spatial_) {
      // add eps to avoid overflow
      norm_data[n] = pow(static_cast<float>(caffe_cpu_asum<Ftype>(dim, buffer_data)+eps_),
                        0.5);
      caffe_cpu_scale<Ftype>(dim, Ftype(1.0 / norm_data[n]), bottom_data,
                             top_data);
    } else {
      caffe_cpu_gemv<Ftype>(CblasTrans, channels, spatial_dim, Ftype(1),
                            buffer_data, sum_channel_multiplier, Ftype(1),
                            norm_data);
      // compute norm
      caffe_powx<Ftype>(spatial_dim, norm_data, Ftype(0.5), norm_data);
      // scale the layer
      caffe_cpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                            1, Ftype(1), sum_channel_multiplier, norm_data,
                            Ftype(0), buffer_data);
      caffe_div<Ftype>(dim, bottom_data, buffer_data, top_data);
      norm_data += spatial_dim;
    }
    // scale the output
    if (channel_shared_) {
      caffe_scal<Ftype>(dim, scale[0], top_data);
    } else {
      caffe_cpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                            1, Ftype(1), scale, sum_spatial_multiplier,
                            Ftype(0),
                            buffer_data);
      caffe_mul<Ftype>(dim, top_data, buffer_data, top_data);
    }
    bottom_data += dim;
    top_data += dim;
  }
}

template <typename Ftype, typename Btype>
void NormalizeLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  const Btype* top_diff = top[0]->cpu_diff<Btype>();
  const Btype* top_data = top[0]->cpu_data<Btype>();
  const Btype* bottom_data = bottom[0]->template cpu_data<Btype>();
  Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
  const Btype* scale = this->blobs_[0]->template cpu_data<Btype>();
  const Btype* norm_data = norm_.template cpu_data<Btype>();
  Btype* buffer_data = buffer_.template mutable_cpu_data<Btype>();
  Btype* buffer_channel = buffer_channel_.template mutable_cpu_data<Btype>();
  Btype* buffer_spatial = buffer_spatial_.template mutable_cpu_data<Btype>();
  const Btype* sum_channel_multiplier = sum_channel_multiplier_.template cpu_data<Btype>();
  const Btype* sum_spatial_multiplier = sum_spatial_multiplier_.template cpu_data<Btype>();
  int count = top[0]->count();
  int num = top[0]->num();
  int dim = count / num;
  int spatial_dim = top[0]->height() * top[0]->width();
  int channels = top[0]->channels();

  // Propagate to param
  if (this->param_propagate_down_[0]) {
    Btype* scale_diff = this->blobs_[0]->template mutable_cpu_diff<Btype>();
    if (channel_shared_) {
      scale_diff[0] +=
          caffe_cpu_dot<Btype>(count, top_data, top_diff) / scale[0];
    } else {
      for (int n = 0; n < num; ++n) {
        caffe_mul<Btype>(dim, top_data+n*dim, top_diff+n*dim, buffer_data);
        caffe_cpu_gemv<Btype>(CblasNoTrans, channels, spatial_dim, Btype(1),
                              buffer_data, sum_spatial_multiplier, Btype(0),
                              buffer_channel);
        // store a / scale[i] in buffer_data temporary
        caffe_div<Btype>(channels, buffer_channel, scale, buffer_channel);
        caffe_add<Btype>(channels, buffer_channel, scale_diff, scale_diff);
      }
    }
  }

  // Propagate to bottom
  if (propagate_down[0]) {
    for (int n = 0; n < num; ++n) {
      if (across_spatial_) {
        Btype a = caffe_cpu_dot<Btype>(dim, bottom_data, top_diff);
        caffe_cpu_scale<Btype>(dim, a / norm_data[n] / norm_data[n],
                               bottom_data, bottom_diff);
        caffe_sub<Btype>(dim, top_diff, bottom_diff, bottom_diff);
        caffe_scal<Btype>(dim, Btype(1.0 / norm_data[n]), bottom_diff);
      } else {
        // dot product between bottom_data and top_diff
        caffe_mul<Btype>(dim, bottom_data, top_diff, buffer_data);
        caffe_cpu_gemv<Btype>(CblasTrans, channels, spatial_dim, Btype(1),
                              buffer_data, sum_channel_multiplier, Btype(0),
                              buffer_spatial);
        // scale bottom_diff
        caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                              1, Btype(1), sum_channel_multiplier,
                              buffer_spatial, Btype(0), buffer_data);
        caffe_mul<Btype>(dim, bottom_data, buffer_data, bottom_diff);
        // divide by square of norm
        caffe_powx<Btype>(spatial_dim, norm_data, Btype(2), buffer_spatial);
        caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                              1, Btype(1), sum_channel_multiplier,
                              buffer_spatial, Btype(0), buffer_data);
        caffe_div<Btype>(dim, bottom_diff, buffer_data, bottom_diff);
        // subtract
        caffe_sub<Btype>(dim, top_diff, bottom_diff, bottom_diff);
        // divide by norm
        caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                              1, Btype(1), sum_channel_multiplier, norm_data,
                              Btype(0), buffer_data);
        caffe_div<Btype>(dim, bottom_diff, buffer_data, bottom_diff);
        norm_data += spatial_dim;
      }
      // scale the diff
      if (channel_shared_) {
        caffe_scal<Btype>(dim, scale[0], bottom_diff);
      } else {
        caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                              1, Btype(1), scale, sum_spatial_multiplier,
                              Btype(0), buffer_data);
        caffe_mul<Btype>(dim, bottom_diff, buffer_data, bottom_diff);
      }
      bottom_data += dim;
      top_diff += dim;
      bottom_diff += dim;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(NormalizeLayer);
#endif

INSTANTIATE_CLASS_FB(NormalizeLayer);
REGISTER_LAYER_CLASS(Normalize);

}  // namespace caffe
