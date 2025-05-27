
CXX = g++

AS = nasm
ASFLAGS = -felf64 -Fdwarf -g
LDFLAGS = -lm
CFLAGS= -Wall -Wextra -pedantic -Og -g -Wno-unused-variable -Wno-unused-parameter -no-pie -z noexecstack

CAMARA_CXXFLAGS = `pkg-config --cflags opencv4` -std=c++17 $(CFLAGS)
CAMARA_LDFLAGS = `pkg-config --libs opencv4`

generate: generate.cpp classifier.cpp forest.cpp
	$(CXX) $(CAMARA_CXXFLAGS) generate.cpp classifier.cpp forest.cpp -o generate_dataset $(CAMARA_LDFLAGS)
	./generate_dataset

train:
	poetry run python train.py 

camara: camara.cpp classifier.cpp forest.cpp
	$(CXX) $(CAMARA_CXXFLAGS) camara.cpp classifier.cpp forest.cpp -o camara $(CAMARA_LDFLAGS)

run: camara
	./camara

pipeline:
	@echo "Iniciando pipeline..."
	make clean
	make generate
	make train
	make camara
	make run
	@echo "Pipeline completado."

clean:
	rm -f camara generate_dataset
	@echo "Clean completado."

.PHONY: test
