#include "Definitions.h"
#include <unsupported/Eigen/KroneckerProduct>

#define kp KroneckerProductSparse<sparseMat, sparseMat>

typedef Eigen::Triplet<scalarType> trip;
typedef Eigen::Matrix<scalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    rmMatrixX_t;

using namespace Eigen;

const int d = 2;                           // size of single-site Hilbert space
sparseMat sigmaplus(d, d),
          sigmaminus(d, d),
          sigmaz(d, d);

sparseMat id(int size)                             // return an identity matrix
{
    sparseMat id(size, size);
    id.reserve(VectorXd::Constant(size, 1));
    for(int i = 0; i < size; i++)
        id.insert(i, i) = 1.;
    return id;
};

sparseMat createNthHeis(int dist)       // TODO: add support for external field
{
    if(dist == 1)
    {
        sparseMat nnHeis(d * d, d * d);
        std::vector<trip> nnHeisElementList
            = {trip(0, 0, 1.), trip(1, 1, -1.), trip(2, 1, 2.), trip(1, 2, 2.),
               trip(2, 2, -1.), trip(3, 3, 1.)};
        nnHeis.setFromTriplets(nnHeisElementList.begin(),
                               nnHeisElementList.end());
        return nnHeis;
    }
    else
    {
        sparseMat middleId = id(pow(d, dist - 1));
        return sparseMat(kp(kp(sigmaz, middleId), sigmaz))
               + 2. * (sparseMat(kp(kp(sigmaplus, middleId), sigmaminus))
                       + sparseMat(kp(kp(sigmaminus, middleId), sigmaplus)));
    };
};

double oneSiteExpValue(Matrix<scalarType, d, d> op, int site, rmMatrixX_t psi,
                       int lSys)
{
    int psiDim = psi.size();
    if(site == 0)
    {
        psi.resize(d, psiDim / d);
        return re((op * psi * psi.adjoint()).trace());
    }
    else if(site == lSys - 1)
    {
        psi.resize(psiDim / d, d);
        return re((psi * op.adjoint() * psi.adjoint()).trace());
    }
    else
    {
        int aSize = pow(d, site),
            cSize = pow(d, lSys - site - 1);
        psi.resize(aSize * d, cSize);
        rmMatrixX_t psiPsiDag = psi * psi.adjoint();
        psiPsiDag.resize(aSize * d * aSize, d);
        rmMatrixX_t psiPsiDagO = psiPsiDag * op;
        psiPsiDagO.resize(aSize * d, aSize * d);
        return re(psiPsiDagO.trace());
    };
};

int main()
{
    // ************* Hamiltonian parameters
    const int farthestNeighborCoupling = 2,
              lSys = 12;
    const std::vector<double> j = {0., 1., 2.};
    // strength of 1st, 2nd, etc. nearest neigbor Heisenberg couplings         // TODO: first term must be 0
    const double lancTolerance = 1.e-6;                // allowed Lanczos error
    #define u1symmetry   // does system have U(1) symmetry? If not, comment out
    #ifdef u1symmetry
        const int targetQNum = 4;  // targeted symmetry sector (e.g. total S^z)
        const std::vector<int> oneSiteQNums = {1, -1};              // hbar = 2
    // ************* end Hamiltonian parameters
        std::vector<int> qNumList = oneSiteQNums;
    #endif
    sigmaplus.reserve(VectorXd::Constant(d, 1));
    sigmaplus.insert(0, 1) = 1.;
    sigmaplus.makeCompressed();
    sigmaminus.reserve(VectorXd::Constant(d, 1));
    sigmaminus.insert(1, 0) = 1.;
    sigmaminus.makeCompressed();
    sigmaz.reserve(VectorXd::Constant(d, 1));
    sigmaz.insert(0, 0) = 1.;
    sigmaz.insert(1, 1) = -1.;
    
    // create coupling operators
    std::vector<sparseMat> couplings(farthestNeighborCoupling + 1);
    for(int i = 0, end = j.size(); i < end; i++)
        if(j[i])
            couplings[i] = j[i] * createNthHeis(i);
    
    // create Hamiltonian
    sparseMat ham(d, d);
    for(int site = 0; site < lSys - 1; site++)               // add on new site
    {
        sparseMat tempHam = kp(ham, id(d));
        ham = tempHam;
        for(int couplingDist = 1; couplingDist <= farthestNeighborCoupling;
            couplingDist++)                     // add in couplings to new site
            if(j[couplingDist])
            {
                if(couplingDist == site + 1)
                    ham += couplings[couplingDist];
                else if(couplingDist <= site + 1)
                    ham += kp(id(pow(d, site - couplingDist + 1)),
                              couplings[couplingDist]);
            };
        #ifdef u1symmetry
            std::vector<int> newQNumList;
            newQNumList.reserve(qNumList.size() * d);
            for (int newQNum : oneSiteQNums)
                for(int oldQNum : qNumList)
                    newQNumList.push_back(oldQNum + newQNum);
            qNumList = newQNumList;
        #endif
    };
    
    // run Lanczos on Hamiltonian to find ground state
    std::cout << "Starting Lanczos..." << std::endl;
    #ifdef u1symmetry
        int sectorSize = std::count(qNumList.begin(), qNumList.end(),
                                    targetQNum);
        std::vector<int> sectorPositions;
        sectorPositions.reserve(sectorSize);
        for(auto firstElement = qNumList.begin(),
            qNumListElement = firstElement, end = qNumList.end();
            qNumListElement != end; qNumListElement++)
            if(*qNumListElement == targetQNum)
                sectorPositions.push_back(qNumListElement - firstElement);
        sparseMat sector(sectorSize, sectorSize);
        sector.reserve(VectorX_t::Constant(sectorSize, sectorSize));
        for(int j = 0; j < sectorSize; j++)
            for(int i = 0; i < sectorSize; i++)
                sector.insert(i, j) = ham.coeffRef(sectorPositions[i],
                                                   sectorPositions[j]);
        sector.makeCompressed();
        VectorX_t seed = VectorX_t::Random(sectorSize).normalized();
        double gsEnergy = lanczos(sector, seed, lancTolerance);
        VectorX_t groundState = VectorX_t::Zero(ham.rows());
        for(int i = 0; i < sectorSize; i++)
            groundState(sectorPositions[i]) = seed(i);
    #else
        VectorX_t groundState = VectorX_t::Random(ham.rows()).normalized();
        double gsEnergy = lanczos(ham, groundState, lancTolerance);
    #endif
    std::cout << "Ground state energy density: " << gsEnergy / lSys << std::endl;
    
    // calculate expectation values of one-site observables (e.g. sigma_z):
    Matrix<scalarType, d, d> op;
    op << 1.,  0.,
          0., -1.;
    std::cout << "One-site expectation values:" << std::endl;
    for(int i = 0; i < lSys; i++)
        std::cout << oneSiteExpValue(op, i, groundState, lSys) << " ";
    std::cout << std::endl;
    
    return 0;
};
