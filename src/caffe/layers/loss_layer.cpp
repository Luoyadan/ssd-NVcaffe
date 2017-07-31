#include <algorithm>
#include <vector>

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void LossLayer<Ftype, Btype>::LayerSetUp(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  // LossLayers have a non-zero (1) loss by default.
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(1.F);
  }
}

template <typename Ftype, typename Btype>
void LossLayer<Ftype, Btype>::Reshape(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
}


template <typename Ftype, typename Btype>
Ftype LossLayer<Ftype, Btype>::GetNormalizer_F(
    const LossParameter_NormalizationMode normalization_mode,
    const int outer_num, const int inner_num, const int valid_count) {
  Ftype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Ftype(outer_num * inner_num);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Ftype(outer_num * inner_num);
      } else {
        normalizer = Ftype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Ftype(outer_num);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Ftype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Ftype(1.0), normalizer);
}

template <typename Ftype, typename Btype>
Btype LossLayer<Ftype, Btype>::GetNormalizer_B(
    const LossParameter_NormalizationMode normalization_mode,
    const int outer_num, const int inner_num, const int valid_count) {
  Btype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Btype(outer_num * inner_num);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Btype(outer_num * inner_num);
      } else {
        normalizer = Btype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Btype(outer_num);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Btype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Btype(1.0), normalizer);
}
INSTANTIATE_CLASS_FB(LossLayer);
}  // namespace caffe
