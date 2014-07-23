PROG = ED
CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -std=c++11 -march=native -I ~/Eigen_3.2.0 -llapack
OBJS = Lanczos.o main.o

$(PROG): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(PROG) $(OBJS)

Lanczos.o: Definitions.h

main.o: Definitions.h
