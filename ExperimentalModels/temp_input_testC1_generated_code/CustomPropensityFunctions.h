#ifndef _CUSTOM_PROPENSITY_FUNCTIONS_H_
#define _CUSTOM_PROPENSITY_FUNCTIONS_H_

#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>

using namespace std;

namespace STOCHKIT
{
template<typename _populationVectorType>
double f0(_populationVectorType& x) {
    return (double)21.05*x[0]/(double)(25+x[0]);
}

template<typename _populationVectorType>
double f1(_populationVectorType& x) {
    return (double)x[0]*0.417;
}

template<typename _populationVectorType>
double f2(_populationVectorType& x) {
    return (double)58.35*x[1]/(double)(6.5+x[1]);
}

template<typename _populationVectorType>
double f5(_populationVectorType& x) {
    return (double)std::max(0.0,(0.83+2(0+x[0])*1.0-M0_0))*pow(0.02,4)/(double)(pow(0.02,4)+pow(M0_0,4)));
}

template<typename _populationVectorType>
class CustomPropensityFunctions
{
public:
    static const int NumberOfReactions = 6;
    typedef double (*PropensityMember)(_populationVectorType&);
    std::vector<PropensityMember> propensityFunctions;

    // default constructor
    CustomPropensityFunctions() {
        propensityFunctions.resize(6);
        propensityFunctions[0] = &f0<_populationVectorType>;
        propensityFunctions[1] = &f1<_populationVectorType>;
        propensityFunctions[2] = &f2<_populationVectorType>;
        propensityFunctions[5] = &f5<_populationVectorType>;
    }
};
}
#endif
