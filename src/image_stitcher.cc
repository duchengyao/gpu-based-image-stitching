// Created by s1nh.org.

#include "image_stitcher.h"

#include <thread>
#include <mutex>

void ImageStitcher::SetParams(
    const int& blend_width,
    std::vector<cv::UMat>& undist_xmap_vector,
    std::vector<cv::UMat>& undist_ymap_vector,
    std::vector<cv::UMat>& reproj_xmap_vector,
    std::vector<cv::UMat>& reproj_ymap_vector,
    std::vector<cv::Rect>& projected_image_roi_vect_refined) {

  std::cout << "[SetParams] Setting params..." << std::endl;
  num_img_ = undist_xmap_vector.size();
  warp_mutex_vector_ = std::vector<std::mutex>(num_img_);

  undist_xmap_vector_ = undist_xmap_vector;
  undist_ymap_vector_ = undist_ymap_vector;
  reproj_xmap_vector_ = reproj_xmap_vector;
  reproj_ymap_vector_ = reproj_ymap_vector;
  roi_vect_ = projected_image_roi_vect_refined;

  // Combine two remap operator (For speed up a little)
  final_xmap_vector_ = std::vector<cv::UMat>(undist_ymap_vector.size());
  final_ymap_vector_ = std::vector<cv::UMat>(undist_ymap_vector.size());
  tmp_umat_vect_ = std::vector<cv::UMat>(undist_ymap_vector.size());
  for (size_t img_idx = 0; img_idx < num_img_; ++img_idx) {
    remap(undist_xmap_vector_[img_idx],
          final_xmap_vector_[img_idx],
          reproj_xmap_vector_[img_idx],
          reproj_ymap_vector_[img_idx],
          cv::INTER_LINEAR);
    remap(undist_ymap_vector_[img_idx],
          final_ymap_vector_[img_idx],
          reproj_xmap_vector_[img_idx],
          reproj_ymap_vector_[img_idx],
          cv::INTER_LINEAR);
//    cv::UMat _;
//    undist_xmap_vector[img_idx].copyTo(_);
//    wrap_vec_.push_back(_);//TODO: Use zeros instead of this fake data.
  }
  CreateWeightMap(undist_ymap_vector[0].rows, blend_width);
  std::cout << "[SetParams] Setting params... Done." << std::endl;
}

void ImageStitcher::CreateWeightMap(const int& height, const int& width) {
  std::cout << "[CreateWeightMap] Creating weight map..." << std::endl;

  // TODO: Try CV_16F.
  cv::Mat _l = cv::Mat(height, width, CV_8UC3);
  cv::Mat _r = cv::Mat(height, width, CV_8UC3);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      _l.at<cv::Vec3b>(i, j)[0] =
      _l.at<cv::Vec3b>(i, j)[1] =
      _l.at<cv::Vec3b>(i, j)[2] =
          cv::saturate_cast<uchar>((float) j / (float) width * 255);

      _r.at<cv::Vec3b>(i, j)[0] =
      _r.at<cv::Vec3b>(i, j)[1] =
      _r.at<cv::Vec3b>(i, j)[2] =
          cv::saturate_cast<uchar>((float) (width - j) / (float) width * 255);

    }
  }
  weightMap_.emplace_back(_l.getUMat(cv::ACCESS_READ));
  weightMap_.emplace_back(_r.getUMat(cv::ACCESS_READ));

  cv::imwrite("../results/_weight_map_l.png", weightMap_[0]);
  cv::imwrite("../results/_weight_map_r.png", weightMap_[1]);

  std::cout << "[CreateWeightMap] Creating weight map... Done." << std::endl;
}

void ImageStitcher::WarpImages(
    const int& img_idx,
    const int& fusion_pixel,
    const std::vector<cv::UMat>& image_vector,
    std::vector<std::mutex>& image_mutex_vector,
    std::vector<cv::UMat>& images_warped_with_roi_vector,
    cv::UMat& image_concat_umat) {

  std::cout << "[WarpImages] Warping images " << img_idx << " of " << num_img_ << "..." << std::endl;
  int64_t t0, t1, t2, t3, t4, t5, t6, tn;
  t0 = cv::getTickCount();
  image_mutex_vector[img_idx].lock();

//  remap(image_vector[img_idx],
//        tmp_umat_vect_[img_idx],
//        undist_xmap_vector_[img_idx],
//        undist_ymap_vector_[img_idx],
//        cv::INTER_LINEAR);
  t1 = cv::getTickCount();

  // Must use UMat.
//  remap(tmp_umat_vect_[img_idx],
//        tmp_umat_vect_[img_idx],
//        reproj_xmap_vector_[img_idx],
//        reproj_ymap_vector_[img_idx],
//        cv::INTER_LINEAR);

  // Combine two remap operator (For speed up a little)

  std::cout << "[WarpImages] Remapping " << img_idx << ":" << num_img_ << " ..." << std::endl;
  remap(image_vector[img_idx],
        tmp_umat_vect_[img_idx],
        final_xmap_vector_[img_idx],
        final_ymap_vector_[img_idx],
        cv::INTER_LINEAR);
  std::cout << "[WarpImages] Remapped " << img_idx << ":" << num_img_ << " ..." << std::endl;
  image_mutex_vector[img_idx].unlock();
  t2 = cv::getTickCount();
  t3 = cv::getTickCount();


  // Blend the edge of 2 images.
//  warp_mutex_vector_[img_idx].lock();
//  tmp_umat_vect_[img_idx].copyTo(wrap_vec_[img_idx]);
//  warp_mutex_vector_[img_idx].unlock();

  if (img_idx > 0) {
    std::cout << "[test 1] "
              << img_idx << ": "
              << roi_vect_[img_idx].x << ", "
              << roi_vect_[img_idx].y << ", "
              << weightMap_[0].cols << ", "
              << weightMap_[0].rows << std::endl;

    cv::UMat _r = tmp_umat_vect_[img_idx](cv::Rect(
        roi_vect_[img_idx].x,
        roi_vect_[img_idx].y,
        weightMap_[0].cols,
        weightMap_[0].rows));

    warp_mutex_vector_[img_idx - 1].lock();

    std::cout << "[test 2] "
              << img_idx << ": "
              << roi_vect_[img_idx - 1].x + roi_vect_[img_idx - 1].width << ", "
              << roi_vect_[img_idx - 1].y << ", "
              << weightMap_[0].cols << ", "
              << weightMap_[0].rows << std::endl;

    cv::UMat _l = tmp_umat_vect_[img_idx - 1](cv::Rect(
        roi_vect_[img_idx - 1].x + roi_vect_[img_idx - 1].width,
        roi_vect_[img_idx - 1].y,
        weightMap_[0].cols,
        weightMap_[0].rows));
    warp_mutex_vector_[img_idx - 1].unlock();

    cv::multiply(_r, weightMap_[0], _r, 1. / 255.);
    cv::multiply(_l, weightMap_[1], _l, 1. / 255.);
    cv::add(_r, _l, _r);
  }

  // Apply ROI.
  int cols = 0;
  for (size_t i = 0; i < img_idx; i++) {
    cols += roi_vect_[i].width;
  }

  std::cout << "[test 3] "
            << img_idx << ": "
            << cols << ", "
            << roi_vect_[img_idx].width << ", "
            << roi_vect_[img_idx].height << std::endl;

  tmp_umat_vect_[img_idx](roi_vect_[img_idx]).copyTo(
      image_concat_umat(cv::Rect(cols, 0, roi_vect_[img_idx].width, roi_vect_[img_idx].height))
  );

  tn = cv::getTickCount();
  std::cout << "[WarpImages] Warped images " << img_idx << " of " << num_img_ << ". ("
            << double(t1 - t0) / cv::getTickFrequency() << ";"
            << double(t2 - t1) / cv::getTickFrequency() << ";"
            << double(t3 - t2) / cv::getTickFrequency() << ";"
            << 1. / double(tn - t0) * cv::getTickFrequency() << ")"
            << std::endl;
}
