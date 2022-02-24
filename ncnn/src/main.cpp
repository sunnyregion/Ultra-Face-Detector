//
//  main.cpp
//  UltraFaceTest
//
//  Created by vealocia on 2019/10/17.
//  Copyright © 2019 vealocia. All rights reserved.
//

#include "UltraFace.hpp"
#include "GenderAge.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "FaceDetector.h"

using namespace std;
int main(int argc, char **argv) {
    if (argc <= 3) {
        fprintf(stderr, "Usage: %s <ncnn bin> <ncnn param> [image files...]\n", argv[0]);
        return 1;
    }

    std::string bin_path = argv[1];
    std::string param_path = argv[2];
    UltraFace ultraface(bin_path, param_path, 320, 240, 1, 0.7); // config model input

    auto bin= "../data/genderage.bin";
    auto param = "../data/genderage.param";
    // std::string bin = argv[3];
    // std::string param = argv[4];
    GenderAge genderface(bin,param);
    
    for (int i = 3; i < argc; i++) {
        std::string image_file = argv[i];
        std::cout << "Processing " << image_file << std::endl;

        cv::Mat frame = cv::imread(image_file);
        ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);

        std::vector<FaceInfo> face_info;
        ultraface.detect(inmat, face_info);

        
        for (int i = 0; i < face_info.size(); i++) {
            auto face = face_info[i];
            cv::Point pt1(face.x1, face.y1);
            cv::Point pt2(face.x2, face.y2);
            // cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
            cv::Rect roi(face.x1, face.y1, face.x2-face.x1, face.y2-face.y1);
            // cv::Mat image_roi = frame(roi);
            
            // genderface.detect(frame,gender_age_info);
            // std::cout << "--------------性别："<<gender_age_info[i].gender<<"\t 年龄："<<gender_age_info[i].age<<"------------"<<std::endl;
            // cv::imwrite("./hello.jpg",image_roi);
        }

        Timer timer;
        string param = "../data/face1m.param";
        string bin = "../data/face1m.bin";
        // const int max_side = 320;

        // slim or RFB
        Detector detector(param, bin, false);

        std::vector<bbox> face_boxes;

        timer.tic();

        detector.Detect(frame, face_boxes);
        timer.toc("----total timer:");

        // float long_side = std::max(frame.cols, frame.rows);
        // float scale = max_side/long_side;
        std::vector<GenderAgeInfo> gender_age_info;
        float scale=1.0;
        for (int j = 0; j < face_boxes.size(); ++j) {
            cv::Rect rect(face_boxes[j].x1/scale, face_boxes[j].y1/scale, face_boxes[j].x2/scale - face_boxes[j].x1/scale, face_boxes[j].y2/scale - face_boxes[j].y1/scale);
            
            cv::Mat image_roi = frame(rect);
            face_box box;
            box.x0 = face_boxes[j].x1;
            box.y0 =  face_boxes[j].y1;
            box.x1 = face_boxes[j].x2;
            box.y1 = face_boxes[j].y2;
            for(int i=0;i<5;i++){
                     box.landmark.x[i]=face_boxes[j].landmark[i]._x;
                     box.landmark.y[i]=face_boxes[j].landmark[i]._y;
            }
            if (face_boxes[j].score>0.9992){
                    // cv::Mat dst_resize; 
                    // cv::resize(image_roi,dst_resize,cv::Size(112,112),0,0);
                    cv::Mat warp = genderface.WarpAffine(frame, &box);
                    
                    genderface.detect(warp,gender_age_info);
                    std::cout << "--------------性别："<<gender_age_info[j].gender<<"\t 年龄："<<gender_age_info[j].age<<"------------"<<std::endl;
                    cv::imwrite("./hello2.jpg",image_roi);
                    cv::imwrite("./hello3.jpg", warp);

                    cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
                    char test[80];
                    sprintf(test, "%f", face_boxes[j].score);

                    cv::putText(frame, test, cv::Size((face_boxes[j].x1/scale), face_boxes[j].y1/scale), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
                    cv::circle(frame, cv::Point(face_boxes[j].landmark[0]._x / scale, face_boxes[j].landmark[0]._y / scale), 1, cv::Scalar(0, 0, 225), 4);
                    cv::circle(frame, cv::Point(face_boxes[j].landmark[1]._x / scale, face_boxes[j].landmark[1]._y / scale), 1, cv::Scalar(0, 255, 225), 4);
                    cv::circle(frame, cv::Point(face_boxes[j].landmark[2]._x / scale, face_boxes[j].landmark[2]._y / scale), 1, cv::Scalar(255, 0, 225), 4);
                    cv::circle(frame, cv::Point(face_boxes[j].landmark[3]._x / scale, face_boxes[j].landmark[3]._y / scale), 1, cv::Scalar(0, 255, 0), 4);
                    cv::circle(frame, cv::Point(face_boxes[j].landmark[4]._x / scale, face_boxes[j].landmark[4]._y / scale), 1, cv::Scalar(255, 0, 0), 4);

            }
            
            
           
           
             
        }

        
        cv::imshow("UltraFace", frame);
        cv::waitKey();
        cv::imwrite("./result.jpg", frame);
    }
    return 0;
}
