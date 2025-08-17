#include "classifier.h"

MockClassifier::MockClassifier() {
      
}

std::string MockClassifier::classify(const std::vector<float>& features) {
   return "happy";
}