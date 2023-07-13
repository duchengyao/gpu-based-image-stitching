// Created by s1nh.org.

#ifndef IMAGE_STITCHING_IMAGE_STITCHER_H
#define IMAGE_STITCHING_IMAGE_STITCHER_H

#include "opencv2/opencv.hpp"

class ImageStitcher {

public:
  void SetParams(const int& blend_width,
                 std::vector<cv::UMat>& undist_xmap_vector,
                 std::vector<cv::UMat>& undist_ymap_vector,
                 std::vector<cv::UMat>& reproj_xmap_vector,
                 std::vector<cv::UMat>& reproj_ymap_vector,
                 std::vector<cv::Rect>& projected_image_roi_vect_refined);

  void WarpImages(const int& img_idx,
                  const int& fusion_pixel,
                  const std::vector<cv::UMat>& image_vector,
                  std::vector<std::mutex>& image_mutex_vector,
                  std::vector<cv::UMat>& images_warped_with_roi_vector,
                  cv::UMat& image_concat_umat);

private:
  size_t num_img_;

//    cv::UMat warp_tmp_l_;
  std::vector<cv::UMat> reproj_xmap_vector_, reproj_ymap_vector_;
  std::vector<cv::UMat> undist_xmap_vector_, undist_ymap_vector_;
  std::vector<cv::UMat> final_xmap_vector_, final_ymap_vector_;
  std::vector<cv::UMat> tmp_umat_vect_;
//    std::vector<cv::UMat> wrap_vec_;
  std::vector<std::mutex> warp_mutex_vector_;
  std::vector<cv::Rect> roi_vect_;
  std::vector<cv::UMat> weightMap_;

  void CreateWeightMap(const int& height, const int& width);

};

#endif //IMAGE_STITCHING_IMAGE_STITCHER_H
