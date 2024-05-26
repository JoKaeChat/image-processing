#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

void Deconvolution(const Mat& inputImg, Mat& outputImg, const Mat& hw)
{
    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI, DFT_SCALE);
    
    Mat planesH[2] = { Mat_<float>(hw.clone()), Mat::zeros(hw.size(), CV_32F) };
    Mat complexH;
    merge(planesH, 2, complexH);
    Mat complexIH;
    mulSpectrums(complexI, complexH, complexIH, 0);
    
    idft(complexIH, complexIH);
    split(complexIH, planes);
    outputImg = planes[0];
}

Mat weinerFilter(Mat &src ,Mat &kernel , double K){
   
    Mat srcDFT, kernelDFT,  channels[2], res, mag, tmp;
    
    dft(src, srcDFT, DFT_COMPLEX_OUTPUT);
    dft(kernel, kernelDFT, DFT_COMPLEX_OUTPUT);
    
    split(kernelDFT, channels);
    magnitude(channels[0], channels[1], mag);
    mag = mag.mul(mag) + K;
    
    
    divide(channels[0], mag, channels[0]);
    divide(channels[1], mag, channels[1]);
    merge(channels, 2, tmp);

    mulSpectrums(srcDFT, tmp, res, 0);
    
    dft(res, res, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
    return res;
}


int main() {
    
    Mat src32,src32_2 , weiner, kernel32;
    
    Mat src = imread("/Users/jdg/Desktop/영상처리 강의자료/Wiener_input1.png", IMREAD_GRAYSCALE);
    Mat src2 = imread("/Users/jdg/Desktop/영상처리 강의자료/Wiener_input2.png", IMREAD_GRAYSCALE);
    Mat kernel = imread("/Users/jdg/Desktop/영상처리 강의자료/Wiener_Kernel.png",IMREAD_GRAYSCALE);
    
    src.convertTo(src32, CV_32F, 1.0 / 255.0);
    src2.convertTo(src32_2, CV_32F, 1.0 / 255.0);
    kernel.convertTo(kernel32, CV_32F, 1.0 / 255.0);
    kernel32 /= sum(kernel32)[0];

    Mat res1, res2;
    res1 = weinerFilter(src32, kernel32, 0.1);
    res2 = weinerFilter(src32_2, kernel32, 0.15);
    
    imshow ("original", src32);
    imshow("result", res1);
    imshow("result2",res2);
    
    waitKey(0);
    return 0;
}

