
CXX = g++

AS = nasm
ASFLAGS = -felf64 -Fdwarf -g
LDFLAGS = -lm
CFLAGS= -Wall -Wextra -pedantic -Og -g -Wno-unused-variable -Wno-unused-parameter -no-pie -z noexecstack

CAMARA_CXXFLAGS = `pkg-config --cflags opencv4` -std=c++17 $(CFLAGS)
CAMARA_LDFLAGS = `pkg-config --libs opencv4`

generate: generate.cpp classifier.cpp
	$(CXX) $(CAMARA_CXXFLAGS) generate.cpp classifier.cpp -o generate_points $(CAMARA_LDFLAGS)
	./generate_points

camara: camara.cpp classifier.cpp
	$(CXX) $(CAMARA_CXXFLAGS) camara.cpp classifier.cpp -o camara $(CAMARA_LDFLAGS)
 
run: camara
	./camara

pipeline:
	@echo "Iniciando pipeline..."
	make clean
	make generate
	make camara
	make run
	@echo "Pipeline completado."

clean:
	rm -f camara generate_points
	@echo "Clean completado."

.PHONY: test
