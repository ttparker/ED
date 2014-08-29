#include "Definitions.h"

extern "C"
{
    void dstemr_(char* JOBZ, char* RANGE, int* N, double* D, double* E,
                 double* VL, double* VU, int* IL, int* IU, int* M, double* W,
                 double* Z, int* LDZ, int* NZC, int* ISUPPZ, bool* TRYRAC,
                 double* WORK, int* LWORK, int* IWORK, int* LIWORK, int* INFO);
};

using namespace Eigen;

double lanczos(const sparseMat& mat, VectorX_t& seed, double lancTolerance)
{
    std::ofstream fileout("LancInfo");
    fileout << "Entered the Lanczos" << std::endl;
    const int globalMinLancIters = 3,
              globalMaxLancIters = 100;
    const double fallbackLancTolerance = 1.e-4;
    int matSize = mat.rows();
    if(matSize == 1)
        return re(sparseMat::InnerIterator(mat, 0).value());
    const int minIters = std::min(matSize, globalMinLancIters),
              maxIters = std::min(matSize, globalMaxLancIters);
    std::vector<double> a,
                        b;
    a.reserve(minIters);
    b.reserve(minIters);
    VectorX_t x = seed;
    Matrix<scalarType, Dynamic, Dynamic> basisVecs = x;
    x.noalias() = mat * basisVecs;
    a.push_back(re(seed.dot(x)));
    b.push_back(0.);
    VectorX_t oldGS;
    int i = 0;                                             // iteration counter
    char JOBZ = 'V',                                 // define dstemr arguments
         RANGE = 'I';
    int N = 1;
    std::vector<double> D,
                        E;
    D.reserve(minIters);
    E.reserve(minIters);
    double VL,
           VU;
    int IL = 1,
        IU = 1,
        M;
    std::vector<double> W;
    W.reserve(minIters);
    VectorXd Z;
    int LDZ,
        NZC = 1;
    std::vector<int> ISUPPZ;
    ISUPPZ.reserve(2);
    bool TRYRAC = true;
    double optLWORK;
    std::vector<double> WORK;
    int LWORK,
        optLIWORK;
    std::vector<int> IWORK;
    int LIWORK,
        INFO;
    double gStateDiff;
          // change in ground state vector across subsequent Lanczos iterations
    do
    {
        fileout << "Starting Lanczos iteration " << N << std::endl;
        i++;
        oldGS = seed;
        
        // Lanczos stage 1: Lanczos iteration
        fileout << "Starting Lanczos stage 1..." << std::endl;
        x -= a[i - 1] * basisVecs.col(i - 1);
        b.push_back(x.norm());
        basisVecs.conservativeResize(NoChange, i + 1);
        basisVecs.col(i) = x / b[i];
        x.noalias() = mat * basisVecs.col(i) - b[i] * basisVecs.col(i - 1);
        a.push_back(re(basisVecs.col(i).dot(x)));
        
        // Lanczos stage 2: diagonalize tridiagonal matrix
        fileout << "Starting Lanczos stage 2..." << std::endl;
        N++;
        D = a;
        E.reserve(N);
        E.assign(b.begin() + 1, b.end());
        W.reserve(N);
        Z.resize(N);
        LDZ = N;
        LWORK = -1;
        LIWORK = -1;
        dstemr_(&JOBZ, &RANGE, &N, D.data(), E.data(), &VL, &VU, &IL, &IU, &M,
                W.data(), Z.data(), &LDZ, &NZC, ISUPPZ.data(), &TRYRAC,
                &optLWORK, &LWORK, &optLIWORK, &LIWORK, &INFO);
                                     // query for optimal workspace allocations
        LWORK = int(optLWORK);
        WORK.reserve(LWORK);
        LIWORK = optLIWORK;
        IWORK.reserve(LIWORK);
        dstemr_(&JOBZ, &RANGE, &N, D.data(), E.data(), &VL, &VU, &IL, &IU, &M,
                W.data(), Z.data(), &LDZ, &NZC, ISUPPZ.data(), &TRYRAC,
                WORK.data(), &LWORK, IWORK.data(), &LIWORK, &INFO);
                                                      // calculate ground state
        seed = (basisVecs * Z).normalized();
        gStateDiff = std::abs(1 - std::abs(seed.dot(oldGS)));
    } while(N < minIters || (N < maxIters && gStateDiff > lancTolerance));
    if(N == maxIters && gStateDiff > lancTolerance)
                          // check if last iteration converges to an eigenstate
    {
        double gStateError = std::abs(1 - std::abs(seed.col(0)
                                                   .dot((mat * seed).col(0)
                                                        .normalized())));
        std::cout << "Warning: final Lanczos iteration reached. The inner "
                  << "product of the final approximate ground state and its "
                  << "normalized image differs from 1 by " << gStateError
                  << std::endl;
        if(gStateError > fallbackLancTolerance)
        {
            std::cerr << "Lanczos algorithm failed to converge after "
                      << maxIters << " iterations." << std::endl;
            exit(EXIT_FAILURE);
        };
    };
    fileout << "Lanczos converged after " << N << " iterations" << std::endl;
    return W.front();
};
