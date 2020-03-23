//
// Created by dhaval on 12/18/19.
//

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include<opencv2/opencv.hpp>

#include "hog_visualization.cpp"
#include "task2.cpp"

int main() {
    // TASK 1
    std::string image_name{"/home/dhaval/TDCV/homework2/data/task1/obj1000.jpg"};

    // Loading image for task1
    cv::Mat image{cv::imread(image_name)};

    if (image.empty()) {
        std::cout << "Can't open file." << std::endl;
        return -1;
    }

    // Image operations
    cv::Mat image_gray;
    cv::Mat image_rotate;
    cv::Mat image_flip;

    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    cv::rotate(image, image_rotate, cv::ROTATE_90_CLOCKWISE);
    cv::flip(image, image_flip, -1); // -1 means flip around both axes

    std::vector<float> descriptors, descriptors_gray, descriptors_rotate, descriptors_flip;
    int image_size = 130;

    cv::Mat image_padded(image_size, image_size, image.depth());
    int left = int((image_size - image.cols) / 2);
    int right = image_size - image.cols - left;
    int top = int((image_size - image.rows) / 2);
    int bottom = image_size - image.rows - top;
    cv::copyMakeBorder(image, image_padded, top, bottom, left, right, cv::BORDER_REPLICATE);

    cv::HOGDescriptor hog(image_padded.size(), cv::Size(50, 50), cv::Size(5, 5), cv::Size(25, 25), 8);

    descriptors = hog_descriptors(image_padded);
    visualizeHOG(image_padded, descriptors, hog, 3);
//    visualizeHOG(image_gray, descriptors_gray, hog, 5);
//    visualizeHOG(image_rotate, descriptors_rotate, hog, 5);
//    visualizeHOG(image_flip, descriptors_flip, hog, 5);

    return 0;
}