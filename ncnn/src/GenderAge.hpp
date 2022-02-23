#ifndef GenderAge_hpp
#define GenderAge_hpp

#pragma once
#include "net.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <vector>

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
    ncnn::Net GenderAgeFace;
};
#endif /* GenderAge_hpp */
