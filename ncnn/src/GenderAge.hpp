#ifndef GenderAge_hpp
#define GenderAge_hpp

#pragma once
#include "net.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <vector>
#include "util.hpp"
typedef struct GenderAgeInfo {
    std::string gender;  //性别 female 、 male
    std::string gender_lite; //性别简称 F、 M
    int age;                     //年龄
} GenderAgeInfo;

class GenderAge
{
  public:
    GenderAge(const std::string &bin_path, const std::string &param_path);
     ~GenderAge();
     
    int detect(cv::Mat &img, std::vector<GenderAgeInfo> &face_listi);

  private:
    int get_age(std::vector<double> output);
    cv::Mat WarpAffine(cv::Mat &img, face_box *faceBox);
    cv::Mat similaryTransform(cv::Mat &src, cv::Mat &dst);
    cv::Mat meanAxis0(const cv::Mat &src);
    cv::Mat elementwiseMinus(const cv::Mat &A, const cv::Mat &B);
    int matrixRank(cv::Mat &M);
    cv::Mat varAxis0(const cv::Mat &src);
  private:
    ncnn::Net GenderAgeFace;
    int m_iImageHeight = 112;
    int m_iImageWidth = 112;    
};
#endif /* GenderAge_hpp */
