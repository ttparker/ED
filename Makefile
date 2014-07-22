PROG = ED
CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -std=c++11 -march=native -I ~/Eigen_3.2.0 -llapack

$(PROG): main.cpp
	$(CXX) $(CXXFLAGS) -o $(PROG) main.cpp
