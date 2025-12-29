CXX = g++

AS = nasm
ASFLAGS = -felf64 -Fdwarf -g
LDFLAGS = -lm
CFLAGS= -Wall -Wextra -pedantic -Og -g -Wno-unused-variable -Wno-unused-parameter -no-pie -z noexecstack

CAMARA_CXXFLAGS = `pkg-config --cflags opencv4` -std=c++17 $(CFLAGS)
CAMARA_LDFLAGS = `pkg-config --libs opencv4`

COMMON_SRCS = classifier.cpp linalg.cpp csv_utils.cpp
COMMON_OBJS = linalg_asm.o

clean:
	rm -f camara generate app initialize_vectorizer *.o
	@echo "Clean completado."

download_dataset: 
	uv run python download_dataset.py

linalg_asm.o: linalg_asm.asm
	$(AS) $(ASFLAGS) linalg_asm.asm -o linalg_asm.o

vectorizer: initialize_vectorizer.cpp $(COMMON_SRCS) $(COMMON_OBJS)
	$(CXX) $(CAMARA_CXXFLAGS) initialize_vectorizer.cpp $(COMMON_SRCS) $(COMMON_OBJS) -o initialize_vectorizer $(CAMARA_LDFLAGS)
	LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$(LD_LIBRARY_PATH) 

init_vectorizer:
	./initialize_vectorizer

app: app.cpp $(COMMON_SRCS) $(COMMON_OBJS)
	$(CXX) $(CAMARA_CXXFLAGS) app.cpp $(COMMON_SRCS) $(COMMON_OBJS) -o app $(CAMARA_LDFLAGS)

run: app
	LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$(LD_LIBRARY_PATH) ./app

full_pipeline:
	@echo "Iniciando pipeline..."
	make clean
	make download_dataset
	make initialize_vectorizer
	make run_application
	@echo "Pipeline completado."


.PHONY: test
