CXX = g++

AS = nasm
ASFLAGS = -felf64 -Fdwarf -g
LDFLAGS = -lm
CFLAGS= -Wall -Wextra -pedantic -Og -g -Wno-unused-variable -Wno-unused-parameter -no-pie -z noexecstack

CAMARA_CXXFLAGS = -I. `pkg-config --cflags opencv4` -std=c++17 $(CFLAGS)
CAMARA_LDFLAGS = `pkg-config --libs opencv4`

CLASSIFIERS = classifiers/mock_classifier.cpp
EXTRACTORS = extractors/feature_extractor.cpp extractors/simple_feature_extractor.cpp #extractors/pca_feature_extractor.cpp

download_dataset: 
	python download_dataset.py

extract_features: extract_features.cpp face_classifier.cpp linalg.cpp
	$(CXX) $(CAMARA_CXXFLAGS) extract_features.cpp face_classifier.cpp linalg.cpp -o extract_features $(CAMARA_LDFLAGS)

application: application.cpp $(CLASSIFIERS) $(EXTRACTORS) face_classifier.cpp linalg.cpp
	$(CXX) $(CAMARA_CXXFLAGS) application.cpp $(CLASSIFIERS) $(EXTRACTORS) face_classifier.cpp linalg.cpp -o application $(CAMARA_LDFLAGS)

run_application: application
	./application

pipeline:
	@echo "Iniciando pipeline..."
	make clean
	make download_dataset
	make extract_features
	make application
	make run
	@echo "Pipeline completado."

clean:
	rm -f application extract_features
	@echo "Clean completado."

.PHONY: test
