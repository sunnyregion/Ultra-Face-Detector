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
    ncnn::Mat img_ncnn = ncnn::Mat::from_pixels_resize(img.data,ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows,128 ,128);
    // const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    // const float norm_vals[3] = {0.0078125f, 0.0078125f, 0.0078125f};
    // img_ncnn.substract_mean_normalize(mean_vals, norm_vals);
    ex.input("data", img_ncnn);
    ncnn::Mat img_out;
    ex.extract("fc1", img_out);
    std::vector<double> out;
    for (int i = 0; i < img_out.w; ++i){
        out.push_back(img_out[i]);
    }
    GenderAgeInfo gai;
    
     if (out[0] > out[1]){
         gai.gender="female";
         gai.gender_lite="F";
        // std::cout << "female" << std::endl;
    }else{
        gai.gender="male";
        gai.gender_lite="M";
        // std::cout << "male" << std::endl;
    }
    gai.age=get_age(out);
    face_list.push_back(gai);
    return 0;
}

int GenderAge::get_age(std::vector<double> out){
    int age = 0;
     for (int j=1; j<101; j++)
    {
        age+=(out[2*j]>out[2*j+1]? 0:1);
    }
    return age;
}