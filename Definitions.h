#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#define realHamiltonian

#ifdef realHamiltonian
    #define scalarType double
#else
    #define scalarType std::complex<double>
#endif

typedef Eigen::SparseMatrix<scalarType> sparseMat;
typedef Eigen::Triplet<scalarType> trip;
typedef Eigen::Matrix<scalarType, Eigen::Dynamic, 1> VectorX_t;

double lanczos(const sparseMat& mat, VectorX_t& seed, double lancTolerance);

#endif
