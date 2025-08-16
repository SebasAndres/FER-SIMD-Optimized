CXX = g++

AS = nasm
ASFLAGS = -felf64 -Fdwarf -g
LDFLAGS = -lm
CFLAGS= -Wall -Wextra -pedantic -Og -g -Wno-unused-variable -Wno-unused-parameter -no-pie -z noexecstack

CAMARA_CXXFLAGS = `pkg-config --cflags opencv4` -std=c++17 $(CFLAGS)
CAMARA_LDFLAGS = `pkg-config --libs opencv4`

download_dataset: 
	python download_dataset.py

initialize_vectorizer: initialize_vectorizer.cpp classifier.cpp linalg.cpp
	$(CXX) $(CAMARA_CXXFLAGS) initialize_vectorizer.cpp classifier.cpp linalg.cpp -o initialize_vectorizer $(CAMARA_LDFLAGS)

application: application.cpp classifier.cpp linalg.cpp
	$(CXX) $(CAMARA_CXXFLAGS) application.cpp classifier.cpp linalg.cpp -o application $(CAMARA_LDFLAGS)

run_application: application
	./application

pipeline:
	@echo "Iniciando pipeline..."
	make clean
	make download_dataset
	make initialize_vectorizer
	make application
	make run
	@echo "Pipeline completado."

clean:
	rm -f camara generate
	@echo "Clean completado."

.PHONY: test
