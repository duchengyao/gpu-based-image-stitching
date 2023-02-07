//
// Created by s1nh.org on 2020/12/1.
//

#ifndef IMAGE_STITCHING_IMAGE_STITCHER_H
#define IMAGE_STITCHING_IMAGE_STITCHER_H

#include "opencv2/opencv.hpp"

using namespace std;

class ImageStitcher {


 public:
    void SetParams(
        const int& blend_width,
        vector<cv::UMat>& undist_xmap_vector,
        vector<cv::UMat>& undist_ymap_vector,
        vector<cv::UMat>& reproj_xmap_vector,
        vector<cv::UMat>& reproj_ymap_vector,
        vector<cv::Rect>& projected_image_roi_vect_refined);

    void WarpImages(
        const int& img_idx,
        const int& fusion_pixel,
        const vector<cv::UMat>& image_vector,
        vector<mutex>& image_mutex_vector,
        vector<cv::UMat>& images_warped_with_roi_vector,
        cv::UMat& image_concat_umat);

    void SimpleImageBlender(
        const size_t& fusion_pixel,
        vector<cv::UMat>& img_vect);

 private:
    size_t num_img_;

//    cv::UMat warp_tmp_l_;
    vector<cv::UMat> reproj_xmap_vector_, reproj_ymap_vector_;
    vector<cv::UMat> undist_xmap_vector_, undist_ymap_vector_;
    vector<cv::UMat> final_xmap_vector_, final_ymap_vector_;
    vector<cv::UMat> tmp_umat_vect_;
//    vector<cv::UMat> wrap_vec_;
    vector<mutex> warp_mutex_vector_;
    vector<cv::Rect> roi_vect_;
    vector<cv::UMat> weightMap_;

    void CreateWeightMap(const int& height, const int& width);

};


#endif //IMAGE_STITCHING_IMAGE_STITCHER_H
