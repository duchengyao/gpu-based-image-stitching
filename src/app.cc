// Created by s1nh.org.

#include "app.h"

#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "image_stitcher.h"
#include "stitching_param_generater.h"

App::App() {
  sensor_data_interface_.InitVideoCapture();

  std::vector<cv::UMat> first_image_vector = std::vector<cv::UMat>(sensor_data_interface_.num_img_);
  std::vector<cv::Mat> first_mat_vector = std::vector<cv::Mat>(sensor_data_interface_.num_img_);
  std::vector<cv::UMat> reproj_xmap_vector;
  std::vector<cv::UMat> reproj_ymap_vector;
  std::vector<cv::UMat> undist_xmap_vector;
  std::vector<cv::UMat> undist_ymap_vector;
  std::vector<cv::Rect> image_roi_vect;

  std::vector<std::mutex> image_mutex_vector(sensor_data_interface_.num_img_);
  sensor_data_interface_.get_image_vector(first_image_vector, image_mutex_vector);

  for (size_t i = 0; i < sensor_data_interface_.num_img_; ++i) {
    first_image_vector[i].copyTo(first_mat_vector[i]);
  }

  StitchingParamGenerator stitching_param_generator(first_mat_vector);

  stitching_param_generator.GetReprojParams(undist_xmap_vector,
                                            undist_ymap_vector,
                                            reproj_xmap_vector,
                                            reproj_ymap_vector,
                                            image_roi_vect);

  image_stitcher_.SetParams(100,
                            undist_xmap_vector,
                            undist_ymap_vector,
                            reproj_xmap_vector,
                            reproj_ymap_vector,
                            image_roi_vect);
  total_cols_ = 0;
  for (size_t i = 0; i < sensor_data_interface_.num_img_; ++i) {
    total_cols_ += image_roi_vect[i].width;
  }
  image_concat_umat_ = cv::UMat(image_roi_vect[0].height, total_cols_, CV_8UC3);
}

[[noreturn]] void App::run_stitching() {
  std::vector<cv::UMat> image_vector(sensor_data_interface_.num_img_);
  std::vector<std::mutex> image_mutex_vector(sensor_data_interface_.num_img_);
  std::vector<cv::UMat> images_warped_vector(sensor_data_interface_.num_img_);
  std::thread record_videos_thread(
      &SensorDataInterface::RecordVideos,
      &sensor_data_interface_);

  int64_t t0, t1, t2, t3, tn;

  size_t frame_idx = 0;
  while (true) {
    t0 = cv::getTickCount();

    std::vector<std::thread> warp_thread_vect;
    sensor_data_interface_.get_image_vector(image_vector, image_mutex_vector);
    t1 = cv::getTickCount();

    for (size_t img_idx = 0; img_idx < sensor_data_interface_.num_img_; ++img_idx) {
      warp_thread_vect.emplace_back(
          &ImageStitcher::WarpImages,
          &image_stitcher_,
          img_idx,
          20,
          image_vector,
          std::ref(image_mutex_vector),
          std::ref(images_warped_vector),
          std::ref(image_concat_umat_)
      );
    }
    for (auto& warp_thread : warp_thread_vect) {
      warp_thread.join();
    }
    t2 = cv::getTickCount();

    imwrite("../results/image_concat_umat_" + std::to_string(frame_idx) + ".png",
            image_concat_umat_);

    frame_idx++;
    tn = cv::getTickCount();

    std::cout << "[app] "
              << double(t1 - t0) / cv::getTickFrequency() << ";"
              << double(t2 - t1) / cv::getTickFrequency() << ";"
              << 1 / (double(t2 - t0) / cv::getTickFrequency()) << " FPS; "
              << 1 / (double(tn - t0) / cv::getTickFrequency()) << " Real FPS." << std::endl;

  }
  record_videos_thread.join();

}

int main() {
  App app;
  app.run_stitching();
}
