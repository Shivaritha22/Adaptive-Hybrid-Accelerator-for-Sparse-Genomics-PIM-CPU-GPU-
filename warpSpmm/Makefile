NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_75 -std=c++11
TARGET = warp_matmul
SOURCE = warp_matmul.cu

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET) $(TARGET).exe

.PHONY: clean

