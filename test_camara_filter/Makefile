# Compilador
AS = nasm
CC = gcc
CXX = g++

# Flags
ASFLAGS = -felf64 -Fdwarf -g
CFLAGS = -Wall -Wextra -pedantic -Og -g -Wno-unused-variable -Wno-unused-parameter -no-pie -z noexecstack
LDFLAGS = -lm
CAMARA_CXXFLAGS = `pkg-config --cflags opencv4` -std=c++11
CAMARA_LDFLAGS = `pkg-config --libs opencv4`

all: train camara

valgrind: build
	valgrind --show-reachable=yes --leak-check=full --error-exitcode=1 ./camara \
	&& echo "No se detectaron errores de memoria"

train: train.c optimized_functions.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

optimized_functions.o: optimized_functions.asm
	$(AS) $(ASFLAGS) $^ -o $@

image_processing.o: image_processing.cpp image_processing.h
	$(CXX) $(CAMARA_CXXFLAGS) -c $< -o $@

camara: camara.cpp optimized_functions.o image_processing.o
	$(CXX) $(CAMARA_CXXFLAGS) camara.cpp optimized_functions.o image_processing.o -o camara $(CAMARA_LDFLAGS)

build: camara
	@echo "Build completado."

run: camara
	./camara

clean:
	rm -f train optimized_functions.o camara image_processing.o *.png

.PHONY: all clean run valgrind build