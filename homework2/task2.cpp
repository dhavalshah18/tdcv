//
// Created by dhaval on 12/20/19.
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
#include <glob.h>

#include "hog_descriptors.cpp"

cv::Ptr<cv::ml::DTrees> task2();

cv::Ptr<cv::ml::DTrees> task2() {
    std::vector<cv::String> fn00;
    cv::glob("/home/dhaval/TDCV/homework2/data/task2/train/00/*.jpg", fn00, false);

    std::vector<cv::String> fn01;
    cv::glob("/home/dhaval/TDCV/homework2/data/task2/train/01/*.jpg", fn01, false);

    std::vector<cv::String> fn02;
    cv::glob("/home/dhaval/TDCV/homework2/data/task2/train/02/*.jpg", fn02, false);

    std::vector<cv::String> fn03;
    cv::glob("/home/dhaval/TDCV/homework2/data/task2/train/03/*.jpg", fn03, false);

    std::vector<cv::String> fn04;
    cv::glob("/home/dhaval/TDCV/homework2/data/task2/train/04/*.jpg", fn04, false);

    std::vector<cv::String> fn05;
    cv::glob("/home/dhaval/TDCV/homework2/data/task2/train/05/*.jpg", fn05, false);

    std::vector<std::vector<cv::String>> fn{fn00, fn01, fn02, fn03, fn04, fn05};
    cv::Mat feats, labels;
    cv::Mat image;

    float name = 0.0;
    for (auto i : fn) {

        size_t count = i.size(); //number of png files in images folder

        for (size_t j=0; j<count; j++) {
            image = cv::imread(i[j]);
            std::vector<float> descriptors;
            descriptors = hog_descriptors(image);

            cv::Mat f = cv::Mat(descriptors).reshape(1, 1); // flatten to a single row
            f.convertTo(f, CV_32F);     // ml needs float data
            feats.push_back(f);         // append at bottom

            labels.push_back(name); // an integer, this is, what you get back in the prediction
        }

        name += 1;
    }

    cv::ml::DTrees* tree = cv::ml::DTrees::create();
    tree->setCVFolds(1);
    tree->setMaxCategories(6);
    tree->setMaxDepth(5);
    tree->setMinSampleCount(40);

    cv::ml::TrainData* train_data = cv::ml::TrainData::create(feats, cv::ml::ROW_SAMPLE, labels);
    tree->train(train_data);
    std::cout << "FINISHED TRAINING" << std::endl;

    cv::ml::DTrees::Node* prediction;
    prediction = tree->predict(feats.at<cv::Mat>(1, 200));
    return tree;
}

