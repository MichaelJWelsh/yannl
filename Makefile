COMPILER=g++
FLAGS=-Wall -Werror -pedantic-errors -std=c++11
INPUT=src/activation.cpp src/connection.cpp src/layer.cpp src/matrix.cpp src/network.cpp
OUTPUT=activation.o connection.o layer.o matrix.o network.o
LIB_OUTPUT=libyannl.a
INCLUDE_FILE=*.hpp
LIB_PATH=/usr/local/lib
INCLUDE_PATH=/usr/local/include

install:
	$(COMPILER) -c $(INPUT) $(FLAGS)
	ar rcs $(LIB_OUTPUT) $(OUTPUT)
	cp $(LIB_OUTPUT) $(LIB_PATH)/$(LIB_OUTPUT)
	cp src/$(INCLUDE_FILE) $(INCLUDE_PATH)
	rm -f $(OUTPUT) $(LIB_OUTPUT)
