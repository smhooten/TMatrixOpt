UNAME := $(shell uname)
EIGEN_DIR := $(shell pwd)/eigen

ifeq ($(UNAME), Linux)
# Linux
	CC = g++
	SHARED_FLAG = -shared
else
	ifeq ($(UNAME), Darwin)
# OS X
		CC = g++
		SHARED_FLAG = -dynamiclib
	else
		CC = g++
	endif
endif

CXXFLAGS = -g -Wall -O3 -std=c++11

INCLUDES = -I${EIGEN_DIR}
LDFLAGS = -lpthread

all: Solve.o solve.cpp solve.hpp
	${CC} ${SHARED_FLAG} -fPIC -o Solve.so Solve.o ${LDFLAGS}
	cp Solve.so ../TMatrixOpt/

Solve.o: solve.cpp solve.hpp
	${CC} -c -fPIC solve.cpp ${CXXFLAGS} -o Solve.o ${INCLUDES}

clean:
	rm -f ./*.o
	rm -f ./*.so
	rm -f ../*.so
	rm -f ../*.pyc
