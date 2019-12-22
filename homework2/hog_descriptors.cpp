//
// Created by dhaval on 12/20/19.
//
#import <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

std::vector<float> hog_descriptors(cv::Mat &image);

std::vector<float> hog_descriptors(cv::Mat &image) {
    std::vector<float> descriptors;

    int image_size = 500;

    cv::Mat image_padded(image_size, image_size, image.depth());
    int left = int((image_size - image.cols) / 2);
    int right = image_size - image.cols - left;
    int top = int((image_size - image.rows) / 2);
    int bottom = image_size - image.rows - top;

    if (left < 0 || right < 0 || top < 0 || bottom < 0) {
        std::cout << "Stupid" << std::endl;
    }
    cv::copyMakeBorder(image, image_padded, top, bottom, left, right, cv::BORDER_REPLICATE);

    cv::HOGDescriptor hog(image_padded.size(), cv::Size(50, 50), cv::Size(5, 5), cv::Size(25, 25), 8);

    hog.compute(image_padded, descriptors, cv::Size(8, 8), cv::Size(0, 0));

    return descriptors;
}