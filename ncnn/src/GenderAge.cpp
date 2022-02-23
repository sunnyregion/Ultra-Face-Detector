#include "GenderAge.hpp"

GenderAge::GenderAge(const std::string &bin_path, const std::string &param_path){
    GenderAgeFace.load_param(param_path.data());
    GenderAgeFace.load_model(bin_path.data());
}

GenderAge::~GenderAge(){
    GenderAgeFace.clear();
}

/**
 * @brief 
 * 
 * @param src 
 * @return cv::Mat 
 */
cv::Mat GenderAge::varAxis0(const cv::Mat &src){
    cv::Mat temp_ = elementwiseMinus(src, meanAxis0(src));
    cv::multiply(temp_, temp_, temp_);
    return meanAxis0(temp_);
}

/**
 * @brief  检测人脸的年龄、性别
 * 
 * @param img 
 * @param face_list 
 * @return int 
 */
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



/**
 * @brief 
 * 
 * @param img 
 * @param faceBox 
 * @return cv::Mat 
 */
cv::Mat GenderAge::WarpAffine(cv::Mat &img, face_box *faceBox){
    auto b = get_cur_time();

    float src_def[5][2] = {{30.2946, 51.6963}, {65.5318, 51.5014}, {48.0252, 71.7366}, {33.5493, 92.3655}, {62.7299, 92.2041}};
    int img_size[2] = {m_iImageHeight, m_iImageWidth};

    if (img_size[1] == 112)
    {
        for (int i = 0; i < 5; i++)
        {
            src_def[i][0] += 8.0;
        }
    }

    cv::Mat src(5, 2, CV_32FC1, src_def);
    memcpy(src.data, src_def, 2 * 5 * sizeof(float));

    float landmark[5][2];
    for (int i = 0; i < 5; i++)
    {
        landmark[i][0] = faceBox->landmark.x[i];
        landmark[i][1] = faceBox->landmark.y[i];
    }

    cv::Mat dst(5, 2, CV_32FC1, landmark);
    memcpy(dst.data, landmark, 2 * 5 * sizeof(float));


    cv::Mat M = similaryTransform(dst, src);
    cv::Mat warped;

    cv::Mat M1 = M.rowRange(0, 2);

    cv::warpAffine(img, warped, M1, cv::Size(img_size[0], img_size[1]));

    return warped;
}
/**
 * @brief 
 * 
 * @param src 
 * @return cv::Mat 
 */
cv::Mat GenderAge::meanAxis0(const cv::Mat &src){
    int num = src.rows;
    int dim = src.cols;

    cv::Mat output(1, dim, CV_32F);

    for (int i = 0; i < dim; i++){
        float sum = 0;

        for (int j = 0; j < num; j++)
        {
            sum += src.at<float>(j, i);
        }

        output.at<float>(0, i) = sum / num;
    }

    return output;
}
/**
 * @brief 
 * 
 * @param M 
 * @return int 
 */
int GenderAge::matrixRank(cv::Mat &M){
    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt);
    cv::Mat1b nonZeroSingularValues = w > 0.0001;
    int rank = countNonZero(nonZeroSingularValues);
    return rank;
}
/**
 * @brief 
 * 
 * @param A 
 * @param B 
 * @return cv::Mat 
 */
cv::Mat GenderAge::elementwiseMinus(const cv::Mat &A, const cv::Mat &B){
    cv::Mat output(A.rows, A.cols, A.type());
    assert(B.cols == A.cols);
    if (B.cols == A.cols)
    {
        for (int i = 0; i < A.rows; i++)
        {
            for (int j = 0; j < B.cols; j++)
            {
                output.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(0, j);
            }
        }
    }
    return output;
}

/**
 * @brief 
 * 
 * @param src 
 * @param dst 
 * @return cv::Mat 
 */
cv::Mat GenderAge::similaryTransform(cv::Mat &src, cv::Mat &dst){
    int num = src.rows;
    int dim = src.cols;

    cv::Mat src_mean = meanAxis0(src);
    cv::Mat dst_mean = meanAxis0(dst);

    cv::Mat src_demean = elementwiseMinus(src, src_mean);
    cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);

    cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);

    cv::Mat d(dim, 1, CV_32F);
    d.setTo(1.0f);

    if (cv::determinant(A) < 0)
    {
        d.at<float>(dim - 1, 0) = -1;
    }

    cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
    cv::Mat U, S, V;
    cv::SVD::compute(A, S, U, V);

    // the SVD function in opencv differ from scipy.
    int rank = matrixRank(A);
    if (rank == 0){
        assert(rank == 0);
    }else if (rank == dim - 1){
        if (cv::determinant(U) * cv::determinant(V) > 0){
            T.rowRange(0, dim).colRange(0, dim) = U * V;
        }else{
           
            int s = d.at<float>(dim - 1, 0) = -1;
            d.at<float>(dim - 1, 0) = -1;

            T.rowRange(0, dim).colRange(0, dim) = U * V;
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_ * V; // np.dot(np.diag(d), V.T)
            cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
            cv::Mat C = B.diag(0);
            T.rowRange(0, dim).colRange(0, dim) = U * twp;
            d.at<float>(dim - 1, 0) = s;
        }
    }
    else
    {
        cv::Mat diag_ = cv::Mat::diag(d);
        cv::Mat twp = diag_ * V.t(); // np.dot(np.diag(d), V.T)
        cv::Mat res = U * twp;       // U
        T.rowRange(0, dim).colRange(0, dim) = -U.t() * twp;
    }

    cv::Mat var_ = varAxis0(src_demean);
    float val = cv::sum(var_).val[0];
    cv::Mat res;
    cv::multiply(d, S, res);
    float scale = 1.0 / val * cv::sum(res).val[0];
    T.rowRange(0, dim).colRange(0, dim) = -T.rowRange(0, dim).colRange(0, dim).t();

    cv::Mat temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
    cv::Mat temp2 = src_mean.t();                        // src_mean.T
    cv::Mat temp3 = temp1 * temp2;                       // np.dot(T[:dim, :dim], src_mean.T)
    cv::Mat temp4 = scale * temp3;

    T.rowRange(0, dim).colRange(dim, dim + 1) = -(temp4 - dst_mean.t());
    T.rowRange(0, dim).colRange(0, dim) *= scale;

    return T;
}
