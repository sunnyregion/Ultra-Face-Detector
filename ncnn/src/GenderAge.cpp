#include "GenderAge.hpp"

GenderAge::GenderAge(const std::string &bin_path, const std::string &param_path){
    GenderAgeFace.load_param(param_path.data());
    GenderAgeFace.load_model(bin_path.data());
}

GenderAge::~GenderAge(){
    GenderAgeFace.clear();
}


int GenderAge::detect(cv::Mat &img, std::vector<GenderAgeInfo> &face_list) {
    if (img.empty()) {
        std::cout << "image is empty ,please check!" << std::endl;
        return -1;
    }
    ncnn::Extractor ex = GenderAgeFace.create_extractor();
    // ex.set_light_mode(true);
    // ex.set_num_threads(4);
    ncnn::Mat img_ncnn = ncnn::Mat::from_pixels_resize(img.data,ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows,112 ,112);
    ex.input("data", img_ncnn);
    ncnn::Mat img_out;
    ex.extract("fc1", img_out);
    std::vector<double> out;
    for (int i = 0; i < img_out.w; ++i){
        out.push_back(img_out[i]);
    }
     if (out[0] > out[1]){
        std::cout << "female" << std::endl;
    }else{
        std::cout << "male" << std::endl;
    }
    return 0;
}