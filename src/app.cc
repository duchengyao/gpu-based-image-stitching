//
// Created by s1nh.org.
//

#include "app.h"

#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "image_stitcher.h"
#include "stitching_param_generater.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnreachableCode"
using namespace std;


App::App() {
  sensorDataInterface_.InitVideoCapture(num_img_);

  vector<cv::UMat> first_image_vector = vector<cv::UMat>(num_img_);
  vector<cv::Mat> first_mat_vector = vector<cv::Mat>(num_img_);
  vector<cv::UMat> reproj_xmap_vector;
  vector<cv::UMat> reproj_ymap_vector;
  vector<cv::UMat> undist_xmap_vector;
  vector<cv::UMat> undist_ymap_vector;
  vector<cv::Rect> image_roi_vect;

  vector<mutex> image_mutex_vector(num_img_);
  sensorDataInterface_.get_image_vector(first_image_vector, image_mutex_vector);

  for (size_t i = 0; i < num_img_; ++i) {
    first_image_vector[i].copyTo(first_mat_vector[i]);
  }

  StitchingParamGenerater stitching_param_generater(first_mat_vector);
  stitching_param_generater.GetReprojParams(undist_xmap_vector,
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
  for (size_t i = 0; i < num_img_; ++i) {
    total_cols_ += image_roi_vect[i].width;
  }
  image_concat_umat_ = cv::UMat(image_roi_vect[0].height, total_cols_, CV_8UC3);
}

[[noreturn]] void App::run_stitching() {
  vector<cv::UMat> image_vector(num_img_);
  vector<mutex> image_mutex_vector(num_img_);
  vector<cv::UMat> images_warped_vector(num_img_);
  thread record_videos_thread(
      &SensorDataInterface::RecordVideos,
      &sensorDataInterface_);
  double t0, t1, t2, t3, tn;

  size_t frame_idx = 0;
  while (true) {
    t0 = cv::getTickCount();

    vector<thread> warp_thread_vect;
    sensorDataInterface_.get_image_vector(image_vector, image_mutex_vector);
    t1 = cv::getTickCount();

    for (size_t img_idx = 0; img_idx < num_img_; ++img_idx) {
      warp_thread_vect.emplace_back(
          &ImageStitcher::WarpImages,
          &image_stitcher_,
          img_idx,
          20,
          image_vector,
          ref(image_mutex_vector),
          ref(images_warped_vector),
          ref(image_concat_umat_)
      );
    }
    for (auto& warp_thread : warp_thread_vect) {
      warp_thread.join();
    }
    t2 = cv::getTickCount();


    imwrite("../results/image_concat_umat_" + to_string(frame_idx) + ".png",
            image_concat_umat_);

    frame_idx++;
    tn = cv::getTickCount();

    cout << "[app] "
         << (t1 - t0) / cv::getTickFrequency() << ";"
         << (t2 - t1) / cv::getTickFrequency() << endl;
    cout << 1 / ((t2 - t0) / cv::getTickFrequency()) << " FPS; "
         << 1 / ((tn - t0) / cv::getTickFrequency()) << " Real FPS." << endl;

  }
  record_videos_thread.join();

}


int main() {
  App app;
  app.run_stitching();
}


#pragma clang diagnostic pop
