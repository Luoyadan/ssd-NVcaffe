#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"

namespace caffe {


template<typename Ftype, typename Btype>
AnnotatedDataLayer<Ftype, Btype>::AnnotatedDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Ftype, Btype>(param),
    cache_(param.data_param().cache()),
    shuffle_(param.data_param().shuffle()) {
  sample_only_.store(this->auto_mode_ && this->phase_ == TRAIN);
  init_offsets();
}

template<typename Ftype, typename Btype>
void
AnnotatedDataLayer<Ftype, Btype>::init_offsets() {
  CHECK_EQ(this->transf_num_, this->threads_num());
  CHECK_LE(parser_offsets_.size(), this->transf_num_);
  CHECK_LE(queue_ids_.size(), this->transf_num_);
  parser_offsets_.resize(this->transf_num_);
  queue_ids_.resize(this->transf_num_);
  for (size_t i = 0; i < this->transf_num_; ++i) {
    parser_offsets_[i] = 0;
    queue_ids_[i] = i * this->parsers_num_;
  }
}


template <typename Ftype, typename Btype>
AnnotatedDataLayer<Ftype, Btype>::~AnnotatedDataLayer() {
  this->StopInternalThread();
}


template<typename Ftype, typename Btype>
void
AnnotatedDataLayer<Ftype, Btype>::InitializePrefetch() {
  if (layer_inititialized_flag_.is_set()) {
    return;
  }
  bool init_parent = true;
  if (Caffe::mode() == Caffe::GPU && this->phase_ == TRAIN && this->auto_mode_) {
    // Here we try to optimize memory split between prefetching and convolution.
    // All data and parameter blobs are allocated at this moment.
    // Now let's find out what's left...
    size_t current_parsers_num_ = this->parsers_num_;
    size_t current_transf_num_ = this->threads_num();
    size_t current_queues_num_ = current_parsers_num_ * current_transf_num_;
#ifndef CPU_ONLY
    const size_t batch_bytes = this->prefetch_[0]->bytes(this->is_gpu_transform());
    size_t gpu_bytes, total_memory;
    GPUMemory::GetInfo(&gpu_bytes, &total_memory, true);

    // minimum accross all GPUs
    static std::atomic<size_t> min_gpu_bytes((size_t) -1);
    atomic_minimum(min_gpu_bytes, gpu_bytes);
    P2PManager::dl_bar_wait();
    gpu_bytes = min_gpu_bytes.load();
    bool starving = gpu_bytes * 6UL < total_memory;

    size_t batches_fit = gpu_bytes / batch_bytes;
    size_t total_batches_fit = current_queues_num_ + batches_fit;
#else
    size_t total_batches_fit = current_queues_num_;
    bool starving = false;
#endif
    float ratio = 3.F;
    Net* pnet = this->parent_net();
    if (pnet != nullptr) {
      Solver* psolver = pnet->parent_solver();
      if (psolver != nullptr) {
        if (pnet->layers().size() < 100) {
          ratio = 2.F; // 1:2 for "i/o bound", 1:3 otherwise
        }
      }
    }
    // TODO Respect the number of CPU cores
    const float fit = std::min(16.F, std::floor(total_batches_fit / ratio));  // 16+ -> "ideal" 4x4
    starving = fit <= 1UL || starving;  // enforce 1x1
    current_parsers_num_ = starving ? 1UL : std::min(4UL,
        std::max(1UL, (size_t) std::lround(std::sqrt(fit))));
    if (cache_ && current_parsers_num_ > 1UL) {
      LOG(INFO) << "[" << Caffe::current_device() << "] Reduced parser threads count from "
                << current_parsers_num_ << " to 1 because cache is used";
      current_parsers_num_ = 1UL;
    }
    current_transf_num_ = starving ? 1UL : std::min(4UL,
        std::max(current_transf_num_, (size_t) std::lround(fit / current_parsers_num_)));
    this->RestartAllThreads(current_transf_num_, true, false, Caffe::next_seed());
    this->transf_num_ = this->threads_num();
    this->parsers_num_ = current_parsers_num_;
    this->queues_num_ = this->transf_num_ * this->parsers_num_;
    BasePrefetchingDataLayer<Ftype, Btype>::InitializePrefetch();
    init_parent = false;
    if (current_transf_num_ > 1) {
      this->next_batch_queue();  // 0th already processed
    }
    if (this->parsers_num_ > 1) {
      parser_offsets_[0]++;  // same as above
    }
  }
  if (init_parent) {
    BasePrefetchingDataLayer<Ftype, Btype>::InitializePrefetch();
  }
  this->go();  // kick off new threads if any

  CHECK_EQ(this->threads_num(), this->transf_num_);
  LOG(INFO) << "[" << Caffe::current_device() << "] Parser threads: "
      << this->parsers_num_ << (this->auto_mode_ ? " (auto)" : "");
  LOG(INFO) << "[" << Caffe::current_device() << "] Transformer threads: "
      << this->transf_num_ << (this->auto_mode_ ? " (auto)" : "");
  layer_inititialized_flag_.set();
}

template<typename Ftype, typename Btype>
size_t AnnotatedDataLayer<Ftype, Btype>::queue_id(size_t thread_id) const {
  const size_t qid = queue_ids_[thread_id] + parser_offsets_[thread_id];
  parser_offsets_[thread_id]++;
  if (parser_offsets_[thread_id] >= this->parsers_num_) {
    parser_offsets_[thread_id] = 0UL;
    queue_ids_[thread_id] += this->parsers_num_ * this->threads_num();
  }
  return qid % this->queues_num_;
};


template <typename Ftype, typename Btype>
void AnnotatedDataLayer<Ftype, Btype>::DataLayerSetUp(
  const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const LayerParameter& param = this->layer_param();	  
  const int batch_size = param.data_param().batch_size();
  
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  
  const bool use_gpu_transform = this->is_gpu_transform();
  const bool cache = cache_ && this->phase_ == TRAIN;
  const bool shuffle = cache && shuffle_ && this->phase_ == TRAIN;
  
  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }
  label_map_file_ = anno_data_param.label_map_file();
  // Make sure dimension is consistent within batch.
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  if (transform_param.has_resize_param()) {
    if (transform_param.resize_param().resize_mode() ==
        ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
      CHECK_EQ(batch_size, 1)
        << "Only support batch size of 1 for FIT_SMALL_SIZE.";
    }
  }
  if (Caffe::mode() == Caffe::GPU && this->phase_ == TRAIN && this->auto_mode_) {
    if (!sample_reader_) {
      sample_reader_ = make_shared<AnnodataReader>(param,
          Caffe::solver_count(),
          this->solver_rank_,
          this->parsers_num_,
          this->threads_num(),
          batch_size,
          true,
          false,
          cache,
          shuffle);
    } else if (!reader_) {
      reader_ = make_shared<AnnodataReader>(param,
          Caffe::solver_count(),
          this->solver_rank_,
          this->parsers_num_,
          this->threads_num(),
          batch_size,
          false,
          true,
          cache,
          shuffle);
    } else {
      // still need to run the rest
    }
  } else if (!reader_) {
    reader_ = make_shared<AnnodataReader>(param,
        Caffe::solver_count(),
        this->solver_rank_,
        this->parsers_num_,
        this->threads_num(),
        batch_size,
        false,
        false,
        cache,
        shuffle);
    start_reading();
  }
  // Read a data point, and use it to initialize the top blob.
  
  shared_ptr<AnnotatedDatum> anno_datum = sample_only_ ? sample_reader_->sample() : reader_->sample();
  init_offsets();

  // Reshape top[0] and prefetch_data according to the batch_size.
  // Note: all these reshapings here in load_batch are needed only in case of
  // different datum shapes coming from database.
  
  // transfer shared_ptr to normal pointer to get 'type' member
  AnnotatedDatum* temp_anno_datum = anno_datum.get();
  vector<int> top_shape = this->data_transformers_[0]->InferBlobShape(temp_anno_datum->datum());
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  // used for gpu_data transform
  vector<int> random_vec_shape(1, batch_size * 3);
  LOG(INFO) << "ReshapePrefetch " << top_shape[0] << ", " << top_shape[1] << ", " << top_shape[2]
            << ", " << top_shape[3];
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
	if (use_gpu_transform) {
      this->prefetch_[i]->gpu_transformed_data_->Reshape(top_shape);
      this->prefetch_[i]->random_vec_.Reshape(random_vec_shape);
    }
  }
  
 
  // label
  if (this->output_labels_) {
    has_anno_type_ = temp_anno_datum->has_type() || anno_data_param.has_anno_type();
    vector<int> label_shape(4, 1);
    if (has_anno_type_) {
      anno_type_ = temp_anno_datum->type();
      if (anno_data_param.has_anno_type()) {
        // If anno_type is provided in AnnotatedDataParameter, replace
        // the type stored in each individual AnnotatedDatum.
        LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
        anno_type_ = anno_data_param.anno_type();
      }
      // Infer the label shape from datum.AnnotationGroup().
      int num_bboxes = 0;
      if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
        for (int g = 0; g < temp_anno_datum->annotation_group_size(); ++g) {
          num_bboxes += temp_anno_datum->annotation_group(g).annotation_size();
        }
        label_shape[0] = 1;
        label_shape[1] = 1;
        // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
        // cpu_data and gpu_data for consistent prefetch thread. Thus we make
        // sure there is at least one bbox.
        label_shape[2] = std::max(num_bboxes, 1);
        label_shape[3] = 8;
      } else {
        LOG(FATAL) << "Unknown annotation type.";
      }
    } else {
      label_shape[0] = batch_size;
    }
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
    }
  LOG(INFO) << "Output data size: " << top[0]->num() << ", " << top[0]->channels() << ", "
            << top[0]->height() << ", " << top[0]->width();


  }
}

// This function is called on prefetch thread
template <typename Ftype, typename Btype>
void AnnotatedDataLayer<Ftype, Btype>::load_batch(Batch<Ftype>* batch, int thread_id, size_t queue_id) {
  const bool sample_only = sample_only_.load();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  // use for recording shape 
  TBlob<Ftype> transformed_blob;
  transformed_blob.Reshape(1,1,1,8);
  if (!sample_only && !reader_) {
    this->DataLayerSetUp(this->bottom_init_, this->top_init_);
  }
  const bool use_gpu_transform = this->is_gpu_transform();
  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const size_t qid = sample_only ? 0UL : queue_id;
  AnnodataReader* reader = sample_only ? sample_reader_.get() : reader_.get();
  shared_ptr<AnnotatedDatum> anno_datum = reader->full_peek(qid);
  //CHECK(anno_datum);
  // transfer shared_ptr to normal pointer to get 'type' member
  AnnotatedDatum* temp_anno_datum = anno_datum.get();
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformers_[thread_id]->InferBlobShape(temp_anno_datum->datum(),
      use_gpu_transform);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  transformed_blob.Reshape(top_shape);
  batch->data_.Reshape(top_shape);
  if (use_gpu_transform) {
    top_shape = this->data_transformers_[thread_id]->InferBlobShape(temp_anno_datum->datum(), false);
    top_shape[0] = batch_size;
    batch->gpu_transformed_data_->Reshape(top_shape);
  }
  size_t out_sizeof_element = 0;
  const bool copy_to_cpu = temp_anno_datum->encoded() || !use_gpu_transform;
  Ftype* top_data = nullptr;
  if (copy_to_cpu) {
    top_data = batch->data_.mutable_cpu_data();
  } else {
#ifndef CPU_ONLY
    top_data = batch->data_.mutable_gpu_data();
#else
    NO_GPU;
#endif
  }
  
  Ftype* top_label = nullptr;
  if (this->output_labels_ && !has_anno_type_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  vector<int> random_vec_shape_(1, batch_size * 3);
  batch->random_vec_.Reshape(random_vec_shape_);
  size_t current_batch_id = 0UL;
  size_t item_id;
  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno;
  int num_bboxes = 0;
  
  for (size_t entry = 0; entry < batch_size; ++entry) {
    anno_datum = reader->full_pop(qid, "Waiting for datum");
	// transfer shared_ptr to normal pointer to get 'type' member
    AnnotatedDatum* temp_anno_datum = anno_datum.get();
    item_id = temp_anno_datum->record_id() % batch_size;
    if (item_id == 0UL) {
      current_batch_id = temp_anno_datum->record_id() / batch_size;
    }
	
    AnnotatedDatum distort_datum;
    AnnotatedDatum* expand_datum = NULL;
    if (transform_param.has_distort_param()) {
      distort_datum.CopyFrom(*temp_anno_datum);
      this->data_transformers_[thread_id]->DistortImage(temp_anno_datum->datum(),
                                            distort_datum.mutable_datum());
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformers_[thread_id]->ExpandImage(distort_datum, expand_datum);
      } else {
        expand_datum = &distort_datum;
      }
    } else {
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformers_[thread_id]->ExpandImage(*temp_anno_datum, expand_datum);
      } else {
        expand_datum = &*temp_anno_datum;
      }
    }
    AnnotatedDatum* sampled_datum = NULL;
    
	
	// sample phase
	bool has_sampled = false;
    if (batch_samplers_.size() > 0) {
      // Generate sampled bboxes from expand_datum.
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
      if (sampled_bboxes.size() > 0) {
        // Randomly pick a sampled bbox and crop the expand_datum.
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
        sampled_datum = new AnnotatedDatum();
        this->data_transformers_[thread_id]->CropImage(*expand_datum,
                                           sampled_bboxes[rand_idx],
                                           sampled_datum);
        has_sampled = true;
      } else {
        sampled_datum = expand_datum;
      }
    } else {
      sampled_datum = expand_datum;
    }
    CHECK(sampled_datum != NULL);
    
    vector<int> shape =
        this->data_transformers_[thread_id]->InferBlobShape(sampled_datum->datum());
    if (transform_param.has_resize_param()) {
      if (transform_param.resize_param().resize_mode() ==
          ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
        transformed_blob.Reshape(shape);
        batch->data_.Reshape(shape);
        top_data = batch->data_.mutable_cpu_data();
      } else {
        CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
              shape.begin() + 1));
      }
    } else {
      CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
            shape.begin() + 1));
    }
    // Apply data transformations (mirror, scale, crop...)
    // Get data offset for this datum to hand off to transform thread
    const size_t offset = batch->data_.offset(item_id);
    Ftype* ptr = top_data + offset; // transformed_data
	Ftype* label_ptr = NULL;
    vector<AnnotationGroup> transformed_anno_vec;
	transformed_blob.set_cpu_data(top_data + offset);
    if (this->output_labels_) {
      if (has_anno_type_) {
        // Make sure all data have same annotation type.
        CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
        if (anno_data_param.has_anno_type()) {
          sampled_datum->set_type(anno_type_);
        } else {
          CHECK_EQ(anno_type_, sampled_datum->type()) <<
              "Different AnnotationType.";
        }
        // Transform datum and annotation_group at the same time
        transformed_anno_vec.clear();
        this->data_transformers_[thread_id]->Transform(*sampled_datum,
                                           transformed_blob,
                                           &transformed_anno_vec);
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
          // Count the number of bboxes.
          for (int g = 0; g < transformed_anno_vec.size(); ++g) {
            num_bboxes += transformed_anno_vec[g].annotation_size();
          }
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
        all_anno[item_id] = transformed_anno_vec;
      } else {
        //this->data_transformers_[thread_id]->Transform(sampled_datum->datum(),
        //                                   ptr);
        // Otherwise, store the label from datum.
        CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
        label_ptr = &top_label[item_id];
      }
    } else {
      //this->data_transformers_[thread_id]->Transform(sampled_datum->datum(),
      //                                   ptr);
    }
	
	if (!this->output_labels_ || !has_anno_type_){
	if (use_gpu_transform) {
      // store the generated random numbers and enqueue the copy
      this->data_transformers_[thread_id]->Fill3Randoms(
          &batch->random_vec_.mutable_cpu_data()[item_id * 3]);
	  //shared_ptr<Datum> datum_ptr(sampled_datum->datum());
	  if (this->output_labels_) {
      *label_ptr = sampled_datum->datum().label();
      }
      this->data_transformers_[thread_id]->Copy(sampled_datum->datum(), ptr, out_sizeof_element);
    } else {
      // Precalculate the necessary random draws so that they are
      // drawn deterministically
      std::array<unsigned int, 3> rand;
      this->data_transformers_[thread_id]->Fill3Randoms(&rand.front());
	  //shared_ptr<Datum> datum_ptr(sampled_datum->datum());
      this->data_transformers_[thread_id]->TransformPtrEntry(sampled_datum->datum(), ptr, rand, this->output_labels_,
          label_ptr);
    }
	 
	}
	
    // clear memory
    if (has_sampled) {
      delete sampled_datum;
    }
    if (transform_param.has_expand_param()) {
      delete expand_datum;
   
    }
    reader->free_push(qid, anno_datum);
  } //end-loop

 

  batch->set_id(current_batch_id);
  sample_only_.store(false);
}

INSTANTIATE_CLASS_FB(AnnotatedDataLayer);
REGISTER_LAYER_CLASS(AnnotatedData);

}  // namespace caffe
