// Created by s1nh.org.

#ifndef IMAGE_STITCHING_STITCHING_PARAM_GENERATER_H
#define IMAGE_STITCHING_STITCHING_PARAM_GENERATER_H

#include "opencv2/opencv.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"

class StitchingParamGenerator {
public:
  explicit StitchingParamGenerator(const std::vector<cv::Mat>& image_vector);

  void GetReprojParams(std::vector<cv::UMat>& undist_xmap_vector,
                       std::vector<cv::UMat>& undist_ymap_vector,
                       std::vector<cv::UMat>& reproj_xmap_vector,
                       std::vector<cv::UMat>& reproj_ymap_vector,
                       std::vector<cv::Rect>& projected_image_roi_refined_vect);

  void InitCameraParam();

  void InitWarper();

  void InitUndistortMap();

private:
  // Default command line args
  std::vector<cv::String> img_names;
  bool try_cuda = false;
  float conf_thresh = 1.f;
  float match_conf = 0.6f;
  std::string matcher_type = "homography";
  std::string estimator_type = "homography";
  std::string ba_cost_func = "reproj";
  std::string ba_refine_mask = "xxxxx";
  cv::detail::WaveCorrectKind wave_correct = cv::detail::WAVE_CORRECT_HORIZ;
  bool save_graph = false;
  std::string save_graph_to;
  std::string warp_type = "spherical";
  int expos_comp_type = cv::detail::ExposureCompensator::GAIN_BLOCKS;
  int expos_comp_nr_feeds = 1;
  int expos_comp_nr_filtering = 2;
  int expos_comp_block_size = 32;
  std::string seam_find_type = "no";
  int blend_type = cv::detail::Blender::MULTI_BAND;
  int timelapse_type = cv::detail::Timelapser::AS_IS;
  float blend_strength = 5;
  std::string result_name = "../results/result.jpg";
  bool timelapse = true;
  int range_width = -1;

  // Variables
  size_t num_img_;
  cv::Point full_image_size_;

  std::vector<cv::Mat> image_vector_;
  std::vector<cv::UMat> mask_vector_, mask_warped_vector_;
  std::vector<cv::Size> image_size_vector_, image_warped_size_vector_;
  std::vector<cv::UMat> reproj_xmap_vector_, reproj_ymap_vector_;
  std::vector<cv::UMat> undist_xmap_vector_, undist_ymap_vector_;

  std::vector<cv::detail::CameraParams> camera_params_vector_;
  std::vector<cv::Rect> projected_image_roi_refined_vect_;
  cv::Ptr<cv::detail::RotationWarper> rotation_warper_;
  cv::Ptr<cv::detail::Timelapser> timelapser_;
  cv::Ptr<cv::detail::Blender> blender_;
};

#endif //IMAGE_STITCHING_STITCHING_PARAM_GENERATER_H

