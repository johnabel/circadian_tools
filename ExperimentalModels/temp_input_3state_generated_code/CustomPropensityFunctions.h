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
    return (double)21.05*x[3]/(double)(25+x[3]);
}

template<typename _populationVectorType>
double f6(_populationVectorType& x) {
    return (double)x[3]*0.417;
}

template<typename _populationVectorType>
double f7(_populationVectorType& x) {
    return (double)58.35*x[4]/(double)(6.5+x[4]);
}

template<typename _populationVectorType>
double f10(_populationVectorType& x) {
    return (double)21.05*x[6]/(double)(25+x[6]);
}

template<typename _populationVectorType>
double f11(_populationVectorType& x) {
    return (double)x[6]*0.417;
}

template<typename _populationVectorType>
double f12(_populationVectorType& x) {
    return (double)58.35*x[7]/(double)(6.5+x[7]);
}

template<typename _populationVectorType>
double f15(_populationVectorType& x) {
    return (double)std::max(0.0,(41.5+2*(0+x[0])*11.0-x[0]))*pow(50,4)/(double)(pow(50,4)+pow(x[0],4));
}

template<typename _populationVectorType>
double f16(_populationVectorType& x) {
    return (double)std::max(0.0,(41.5+2*(0+x[0]+x[3])*10.5-x[3]))*pow(50,4)/(double)(pow(50,4)+pow(x[3],4));
}

template<typename _populationVectorType>
double f17(_populationVectorType& x) {
    return (double)std::max(0.0,(41.5+2*(0+x[3]+x[6])*10.5-x[6]))*pow(50,4)/(double)(pow(50,4)+pow(x[6],4));
}

template<typename _populationVectorType>
class CustomPropensityFunctions
{
public:
    static const int NumberOfReactions = 18;
    typedef double (*PropensityMember)(_populationVectorType&);
    std::vector<PropensityMember> propensityFunctions;

    // default constructor
    CustomPropensityFunctions() {
        propensityFunctions.resize(18);
        propensityFunctions[0] = &f0<_populationVectorType>;
        propensityFunctions[1] = &f1<_populationVectorType>;
        propensityFunctions[2] = &f2<_populationVectorType>;
        propensityFunctions[5] = &f5<_populationVectorType>;
        propensityFunctions[6] = &f6<_populationVectorType>;
        propensityFunctions[7] = &f7<_populationVectorType>;
        propensityFunctions[10] = &f10<_populationVectorType>;
        propensityFunctions[11] = &f11<_populationVectorType>;
        propensityFunctions[12] = &f12<_populationVectorType>;
        propensityFunctions[15] = &f15<_populationVectorType>;
        propensityFunctions[16] = &f16<_populationVectorType>;
        propensityFunctions[17] = &f17<_populationVectorType>;
    }
};
}
#endif
