#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#define realHamiltonian

#ifdef realHamiltonian
    #define scalarType double
    #define re
    #include <cmath>
#else
    #define scalarType std::complex<double>
    #define re std::real
#endif

typedef Eigen::SparseMatrix<scalarType> sparseMat;
typedef Eigen::Matrix<scalarType, Eigen::Dynamic, 1> VectorX_t;

double lanczos(const sparseMat& mat, VectorX_t& seed, double lancTolerance);

#endif
