//
// Created by dhaval on 12/20/19.
//

#include <iostream>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/ml.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>


class RandomForest {
public:
    RandomForest(int num_trees, int cv_folds, int max_categories, int max_depth, int min_sample_count) {
        _num_trees = num_trees;
        _cv_folds = cv_folds;
        _max_categories = max_categories;
        _max_depth = max_depth;
        _min_sample_count = min_sample_count;
    }

    void create() {
        for (int t = 0; t < _num_trees; ++t) {
            cv::ml::DTrees* tree = cv::ml::DTrees::create();
            tree->setCVFolds(_cv_folds);
            tree->setMaxCategories(_max_categories);
            tree->setMaxDepth(_max_depth);
            tree->setMinSampleCount(_min_sample_count);

            _forrrrrrrrest.push_back(tree);
        }
    }

    cv::ml::TrainData* bootstrap() {
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

    }

    void train() {
        for (int t=0; t<_num_trees;++t) {
            cv::ml::TrainData* bootstrap = cv::ml::TrainData::create(feats, cv::ml::ROW_SAMPLE, labels);
            _forrrrrrrrest[t] -> train(bootstrap)
        }
    }

private:
    int _num_trees;
    int _cv_folds;
    int _max_categories;
    int _max_depth;
    int _min_sample_count;
    std::vector<cv::ml::DTrees*> _forrrrrrrrest;
};