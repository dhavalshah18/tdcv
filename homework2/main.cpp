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

//    descriptors = hog_descriptors(image);
//    visualizeHOG(image_padded, descriptors, hog, 5);
//    visualizeHOG(image_gray, descriptors_gray, hog, 5);
//    visualizeHOG(image_rotate, descriptors_rotate, hog, 5);
//    visualizeHOG(image_flip, descriptors_flip, hog, 5);

    auto tree = task2();

    std::string test_image_name{"/home/dhaval/TDCV/homework2/data/task2/test/01/0067.jpg"};

    // Loading image for task1
    cv::Mat image_test{cv::imread(test_image_name)};
    std::vector<float> descriptors1{hog_descriptors(image_test)};

    cv::Mat f = cv::Mat(descriptors1).reshape(1, 1); // flatten to a single row
    f.convertTo(f, CV_32F);
    cv::ml::DTrees::Node* prediction;
    prediction = tree->predict(f);

    return 0;
}