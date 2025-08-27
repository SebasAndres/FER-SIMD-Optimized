#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <iostream>
#include <vector>

class EmotionClassifier {
public:
    EmotionClassifier(){
        // Constructor implementation can be empty or include initialization logic if needed
    }
    virtual ~EmotionClassifier() = default;
    virtual std::string classify(const std::vector<float>& features) = 0;
};


class MockClassifier : public EmotionClassifier {
public:
    MockClassifier();
    std::string classify(const std::vector<float>& features) override;
};


class KNNClassifier : public EmotionClassifier {
public:
    KNNClassifier(const std::string& dataset_path);
    std::string classify(const std::vector<float>& features) override;

private:
    std::vector<std::pair<std::string, std::vector<float>>> points;
};

#endif