
CXX = g++

AS = nasm
ASFLAGS = -felf64 -Fdwarf -g
LDFLAGS = -lm
CFLAGS= -Wall -Wextra -pedantic -Og -g -Wno-unused-variable -Wno-unused-parameter -no-pie -z noexecstack

CAMARA_CXXFLAGS = `pkg-config --cflags opencv4` -std=c++17 $(CFLAGS)
CAMARA_LDFLAGS = `pkg-config --libs opencv4`

test: test.c 
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

main: main.cpp 
	$(CXX) $(CAMARA_CXXFLAGS) main.cpp -o main $(CAMARA_LDFLAGS)

camara: camara.cpp classifier.cpp
	$(CXX) $(CAMARA_CXXFLAGS) camara.cpp classifier.cpp -o camara $(CAMARA_LDFLAGS)

run: camara
	./camara

clean:
	rm -f test main camara
	@echo "Clean completado."

.PHONY: test
