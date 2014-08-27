#include "Definitions.h"
#include <time.h>
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

sparseMat id(int sites)                            // return an identity matrix
{
    int size = pow(d, sites);
    sparseMat id(size, size);
    id.reserve(VectorXd::Constant(size, 1));
    for(int i = 0; i < size; i++)
        id.insert(i, i) = 1.;
    return id;
};

sparseMat createNthCoupling(int dist)
{
    if(dist == 1)
    {
        return sparseMat(kp(sigmaz, sigmaz))
               + 2 * (sparseMat(kp(sigmaplus, sigmaminus))
                      + sparseMat(kp(sigmaminus, sigmaplus)));
    }
    else
    {
        sparseMat middleId = id(dist - 1);
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
    const int farthestNeighborCoupling = 6,
              lSys = 16,
              nSiteTypes = 3;                          // size of lattice basis
    const std::vector<double> couplingConstants = {0., 1., -1., 1.};
    #define h couplingConstants[0]
    #define jprime couplingConstants[1]
    #define j1 couplingConstants[2]
    #define j2 couplingConstants[3]
    Matrix<double, nSiteTypes, farthestNeighborCoupling + 1> j;
     // assigns couplings to each site on basis - the row gives the basis site,
     // the column j gives the jth-nearest-neigbor coupling constant for the
     // bonds connecting to that basis site from behind.  The zeroth element of
     // each row is the external field on that basis site.
    j << h, jprime,     0., j1, 0., 0., j2,
         h,     0., jprime, j1, 0., 0., j2,
         h, jprime, jprime, 0., 0., 0., 0.;         // diamond ladder couplings
    const double lancTolerance = 1.e-9;                // allowed Lanczos error
    #define u1Symmetry        // system have U(1) symmetry? If not, comment out
//    #define externalField
         // system in external field with NO U(1) symmetry? If not, comment out
    #ifdef u1Symmetry
        const int targetQNum = 6;  // targeted symmetry sector (e.g. total S^z)
        const std::vector<int> oneSiteQNums = {1, -1};              // hbar = 2
    // ************* end Hamiltonian parameters
        std::vector<int> qNumList = oneSiteQNums;
    #endif
    
    clock_t start = clock();
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
    std::vector<sparseMat> couplingOperators(farthestNeighborCoupling + 1);
    for(int i = 1; i <= farthestNeighborCoupling; i++)
        couplingOperators[i] = createNthCoupling(i);
    
    // create Hamiltonian
    sparseMat ham(d, d);
    #ifdef externalField
        sparseMat h1(d, d);
        h1.reserve(VectorXd::Constant(d, 1));
        h1.insert(0, 0) = -1;
        h1.insert(1, 1) = 1;
        ham = h1;
    #endif
    for(int site = 0; site < lSys / 2 - 1; site++)           // add on new site
    {
        int thisSiteType = site % nSiteTypes;
        sparseMat tempHam = kp(ham, id(1));
        ham = tempHam;
        #ifdef externalField
            ham += kp(id(site + 1), j(thisSiteType, 0) * h1);
        #endif
        for(int couplingDist = 1; couplingDist <= farthestNeighborCoupling;
            couplingDist++)                     // add in couplings to new site
            if(j(thisSiteType, couplingDist))
            {
                if(couplingDist == site + 1)
                    ham += j(thisSiteType, couplingDist)
                           * couplingOperators[couplingDist];
                else if(couplingDist <= site + 1)
                    ham += kp(id(site - couplingDist + 1),
                              j(thisSiteType, couplingDist)
                              * couplingOperators[couplingDist]);
            };
        
        // create the superblock:
        sparseMat hSuper = kp(ham, id(site + 2))
                           + kp(id(site + 2), ham);
        int nextSiteType = (site + 1) % nSiteTypes;
        sparseMat centerBondCouplings(pow(d, site + 3));
        for(int i = 1; i <= farthestNeighborCoupling; i++)
            if(j(nextSiteType, i))
                centerBondCouplings += j(nextSiteType, i)
                                       * createNthCoupling(site + 3 - i);
        hSuper += kp(id(site + 1), centerBondCouplings);
        #ifdef u1Symmetry
            std::vector<int> newQNumList;
            newQNumList.reserve(qNumList.size() * d);
            for(int oldQNum : qNumList)
                for (int newQNum : oneSiteQNums)
                    newQNumList.push_back(oldQNum + newQNum);
            qNumList = newQNumList;
        #endif
        
    
    // run Lanczos on Hamiltonian to find ground state
    std::cout << "Starting Lanczos..." << std::endl;
    #ifdef u1Symmetry
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
        std::cout << oneSiteExpValue(op, i, groundState, lSys) << std::endl;
    std::cout << std::endl;
    clock_t stop = clock();
    std::cout << "Done. Elapsed time: " << float(stop - start)/CLOCKS_PER_SEC
              << " s" << std::endl;
    
    return 0;
};
