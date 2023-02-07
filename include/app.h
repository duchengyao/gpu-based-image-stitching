//
// Created by s1nh.org on 2020/12/2.
//

#ifndef IMAGE_STITCHING_APP_H
#define IMAGE_STITCHING_APP_H

#include "opencv2/opencv.hpp"

#include "sensor_data_interface.h"
#include "image_stitcher.h"

using namespace std;

class App {
 public:
    App();

    [[noreturn]] void run_stitching();

 private:
    std::size_t num_img_;
    SensorDataInterface sensorDataInterface_;
    ImageStitcher image_stitcher_;
    vector<cv::Mat> image_vector_;
    cv::UMat image_concat_umat_;
    int total_cols_;

};

#endif //IMAGE_STITCHING_APP_H
