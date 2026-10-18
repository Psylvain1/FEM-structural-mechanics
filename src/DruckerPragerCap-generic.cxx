/*!
* \file   DruckerPragerCap-generic.cxx
* \brief  This file implements the umat interface for the DruckerPragerCap behaviour law
* \author Thomas Helfer
* \date   
*/

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif /* NOMINMAX */
#include <windows.h>
#ifdef small
#undef small
#endif /* small */
#endif /* _WIN32 */

#ifndef MFRONT_SHAREDOBJ
#define MFRONT_SHAREDOBJ TFEL_VISIBILITY_EXPORT
#endif /* MFRONT_SHAREDOBJ */

#ifndef MFRONT_EXPORT_SYMBOL
#define MFRONT_EXPORT_SYMBOL(TYPE, NAME, VALUE) \
  MFRONT_SHAREDOBJ extern TYPE NAME;            \
  MFRONT_SHAREDOBJ TYPE NAME = VALUE
#endif /* MFRONT_EXPORT_SYMBOL*/

#ifndef MFRONT_EXPORT_ARRAY_ARGUMENTS
#define MFRONT_EXPORT_ARRAY_ARGUMENTS(...) __VA_ARGS__
#endif /* MFRONT_EXPORT_ARRAY_ARGUMENTS */

#ifndef MFRONT_EXPORT_ARRAY_OF_SYMBOLS
#define MFRONT_EXPORT_ARRAY_OF_SYMBOLS(TYPE, NAME, SIZE, VALUE) \
  MFRONT_SHAREDOBJ extern TYPE NAME[SIZE];                      \
  MFRONT_SHAREDOBJ TYPE NAME[SIZE] = {VALUE}
#endif /* MFRONT_EXPORT_ARRAY_OF_SYMBOLS*/

#include<iostream>
#include<cstdlib>
#include"TFEL/Material/OutOfBoundsPolicy.hxx"
#include"TFEL/Math/t2tot2.hxx"
#include"TFEL/Math/t2tost2.hxx"
#include"TFEL/Material/DruckerPragerCap.hxx"
#include"MFront/GenericBehaviour/GenericBehaviourTraits.hxx"
#include"MFront/GenericBehaviour/Integrate.hxx"
#include"MFront/GenericBehaviour/DruckerPragerCap-generic.hxx"

static tfel::material::OutOfBoundsPolicy&
DruckerPragerCap_getOutOfBoundsPolicy(){
static auto policy = tfel::material::None;
return policy;
}

namespace mfront::gb{

template<>
struct GenericBehaviourTraits<tfel::material::DruckerPragerCap<tfel::material::ModellingHypothesis::AXISYMMETRICALGENERALISEDPLANESTRAIN, real, false>>{
static constexpr auto hypothesis = tfel::material::ModellingHypothesis::AXISYMMETRICALGENERALISEDPLANESTRAIN;
static constexpr auto N = tfel::material::ModellingHypothesisToSpaceDimension<hypothesis>::value;
static constexpr auto StensorSize = tfel::material::ModellingHypothesisToStensorSize<hypothesis>::value;
static constexpr auto TensorSize = tfel::material::ModellingHypothesisToTensorSize<hypothesis>::value;
};

template<>
struct GenericBehaviourTraits<tfel::material::DruckerPragerCap<tfel::material::ModellingHypothesis::AXISYMMETRICALGENERALISEDPLANESTRESS, real, false>>{
static constexpr auto hypothesis = tfel::material::ModellingHypothesis::AXISYMMETRICALGENERALISEDPLANESTRESS;
static constexpr auto N = tfel::material::ModellingHypothesisToSpaceDimension<hypothesis>::value;
static constexpr auto StensorSize = tfel::material::ModellingHypothesisToStensorSize<hypothesis>::value;
static constexpr auto TensorSize = tfel::material::ModellingHypothesisToTensorSize<hypothesis>::value;
};

template<>
struct GenericBehaviourTraits<tfel::material::DruckerPragerCap<tfel::material::ModellingHypothesis::AXISYMMETRICAL, real, false>>{
static constexpr auto hypothesis = tfel::material::ModellingHypothesis::AXISYMMETRICAL;
static constexpr auto N = tfel::material::ModellingHypothesisToSpaceDimension<hypothesis>::value;
static constexpr auto StensorSize = tfel::material::ModellingHypothesisToStensorSize<hypothesis>::value;
static constexpr auto TensorSize = tfel::material::ModellingHypothesisToTensorSize<hypothesis>::value;
};

template<>
struct GenericBehaviourTraits<tfel::material::DruckerPragerCap<tfel::material::ModellingHypothesis::PLANESTRESS, real, false>>{
static constexpr auto hypothesis = tfel::material::ModellingHypothesis::PLANESTRESS;
static constexpr auto N = tfel::material::ModellingHypothesisToSpaceDimension<hypothesis>::value;
static constexpr auto StensorSize = tfel::material::ModellingHypothesisToStensorSize<hypothesis>::value;
static constexpr auto TensorSize = tfel::material::ModellingHypothesisToTensorSize<hypothesis>::value;
};

template<>
struct GenericBehaviourTraits<tfel::material::DruckerPragerCap<tfel::material::ModellingHypothesis::PLANESTRAIN, real, false>>{
static constexpr auto hypothesis = tfel::material::ModellingHypothesis::PLANESTRAIN;
static constexpr auto N = tfel::material::ModellingHypothesisToSpaceDimension<hypothesis>::value;
static constexpr auto StensorSize = tfel::material::ModellingHypothesisToStensorSize<hypothesis>::value;
static constexpr auto TensorSize = tfel::material::ModellingHypothesisToTensorSize<hypothesis>::value;
};

template<>
struct GenericBehaviourTraits<tfel::material::DruckerPragerCap<tfel::material::ModellingHypothesis::GENERALISEDPLANESTRAIN, real, false>>{
static constexpr auto hypothesis = tfel::material::ModellingHypothesis::GENERALISEDPLANESTRAIN;
static constexpr auto N = tfel::material::ModellingHypothesisToSpaceDimension<hypothesis>::value;
static constexpr auto StensorSize = tfel::material::ModellingHypothesisToStensorSize<hypothesis>::value;
static constexpr auto TensorSize = tfel::material::ModellingHypothesisToTensorSize<hypothesis>::value;
};

template<>
struct GenericBehaviourTraits<tfel::material::DruckerPragerCap<tfel::material::ModellingHypothesis::TRIDIMENSIONAL, real, false>>{
static constexpr auto hypothesis = tfel::material::ModellingHypothesis::TRIDIMENSIONAL;
static constexpr auto N = tfel::material::ModellingHypothesisToSpaceDimension<hypothesis>::value;
static constexpr auto StensorSize = tfel::material::ModellingHypothesisToStensorSize<hypothesis>::value;
static constexpr auto TensorSize = tfel::material::ModellingHypothesisToTensorSize<hypothesis>::value;
};

} // end of namespace mfront::gb

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

MFRONT_EXPORT_SYMBOL(const char*, DruckerPragerCap_author, "Thomas Helfer");

MFRONT_EXPORT_SYMBOL(const char*, DruckerPragerCap_date, "");

MFRONT_EXPORT_SYMBOL(const char*, DruckerPragerCap_description, "A simple implementation of a perfect\nplasticity behaviour using the\nDrucker - Prager yield criterion\nclosed by a cap .");

MFRONT_EXPORT_SYMBOL(const char*, DruckerPragerCap_validator, "");

MFRONT_EXPORT_SYMBOL(const char*, DruckerPragerCap_build_id, "");

MFRONT_EXPORT_SYMBOL(const char*, DruckerPragerCap_mfront_ept, "DruckerPragerCap");

MFRONT_EXPORT_SYMBOL(const char*, DruckerPragerCap_tfel_version, "5.0.0-dev");

MFRONT_EXPORT_SYMBOL(const char*, DruckerPragerCap_unit_system, "");

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_mfront_mkt, 1u);

MFRONT_EXPORT_SYMBOL(const char*, DruckerPragerCap_mfront_interface, "Generic");

MFRONT_EXPORT_SYMBOL(const char*, DruckerPragerCap_src, "plasticity.mfront");

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_nModellingHypotheses, 7u);

MFRONT_EXPORT_ARRAY_OF_SYMBOLS(const char *, DruckerPragerCap_ModellingHypotheses, 7, MFRONT_EXPORT_ARRAY_ARGUMENTS("AxisymmetricalGeneralisedPlaneStrain",
"AxisymmetricalGeneralisedPlaneStress","Axisymmetrical","PlaneStress","PlaneStrain","GeneralisedPlaneStrain",
"Tridimensional"));

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_nMainVariables, 1u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_nGradients, 1u);

MFRONT_EXPORT_ARRAY_OF_SYMBOLS(int, DruckerPragerCap_GradientsTypes, 1, MFRONT_EXPORT_ARRAY_ARGUMENTS(1));

MFRONT_EXPORT_ARRAY_OF_SYMBOLS(const char *, DruckerPragerCap_Gradients, 1, MFRONT_EXPORT_ARRAY_ARGUMENTS("Strain"));

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_nThermodynamicForces, 1u);

MFRONT_EXPORT_ARRAY_OF_SYMBOLS(int, DruckerPragerCap_ThermodynamicForcesTypes, 1, MFRONT_EXPORT_ARRAY_ARGUMENTS(1));

MFRONT_EXPORT_ARRAY_OF_SYMBOLS(const char *, DruckerPragerCap_ThermodynamicForces, 1, MFRONT_EXPORT_ARRAY_ARGUMENTS("Stress"));

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_nTangentOperatorBlocks, 2u);

MFRONT_EXPORT_ARRAY_OF_SYMBOLS(const char *, DruckerPragerCap_TangentOperatorBlocks, 2, MFRONT_EXPORT_ARRAY_ARGUMENTS("Stress",
"Strain"));

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_BehaviourType, 1u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_BehaviourKinematic, 1u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_SymmetryType, 0u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_ElasticSymmetryType, 0u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_api_version, 1u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_TemperatureRemovedFromExternalStateVariables, 1u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_UsableInPurelyImplicitResolution, 1u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_nMaterialProperties, 0u);

MFRONT_EXPORT_SYMBOL(const char * const *, DruckerPragerCap_MaterialProperties, nullptr);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_nInternalStateVariables, 3u);
MFRONT_EXPORT_ARRAY_OF_SYMBOLS(const char *, DruckerPragerCap_InternalStateVariables, 3, MFRONT_EXPORT_ARRAY_ARGUMENTS("ElasticStrain",
"EquivalentPlasticStrain[0]","EquivalentPlasticStrain[1]"));

MFRONT_EXPORT_ARRAY_OF_SYMBOLS(int, DruckerPragerCap_InternalStateVariablesTypes, 3, MFRONT_EXPORT_ARRAY_ARGUMENTS(1,0,0));

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_nExternalStateVariables, 0u);
MFRONT_EXPORT_SYMBOL(const char * const *, DruckerPragerCap_ExternalStateVariables, nullptr);

MFRONT_EXPORT_SYMBOL(const int *, DruckerPragerCap_ExternalStateVariablesTypes, nullptr);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_nParameters, 13u);

MFRONT_EXPORT_ARRAY_OF_SYMBOLS(const char *, DruckerPragerCap_Parameters, 13, MFRONT_EXPORT_ARRAY_ARGUMENTS("RelativeValueForTheEquivalentStressLowerBoundDefinition",
"epsilon","theta","YoungModulus","PoissonRatio","d",
"beta","pa","pb","minimal_time_step_scaling_factor","maximal_time_step_scaling_factor",
"numerical_jacobian_epsilon","iterMax"));

MFRONT_EXPORT_ARRAY_OF_SYMBOLS(int, DruckerPragerCap_ParametersTypes, 13, MFRONT_EXPORT_ARRAY_ARGUMENTS(0,0,0,0,0,0,0,0,0,0,0,0,2));

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_RelativeValueForTheEquivalentStressLowerBoundDefinition_ParameterDefaultValue, 1e-12);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_epsilon_ParameterDefaultValue, 1e-16);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_theta_ParameterDefaultValue, 1);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_YoungModulus_ParameterDefaultValue, 150000000000);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_PoissonRatio_ParameterDefaultValue, 0.3);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_d_ParameterDefaultValue, 150000000);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_beta_ParameterDefaultValue, 0.6);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_pa_ParameterDefaultValue, -10000000);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_pb_ParameterDefaultValue, -75000000);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_minimal_time_step_scaling_factor_ParameterDefaultValue, 0.1);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_maximal_time_step_scaling_factor_ParameterDefaultValue, 1.7976931348623e+308);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_numerical_jacobian_epsilon_ParameterDefaultValue, 1e-17);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_iterMax_ParameterDefaultValue, 100);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_requiresStiffnessTensor, 0u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_requiresThermalExpansionCoefficientTensor, 0u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_nInitializeFunctions, 0u);

MFRONT_EXPORT_SYMBOL(const char * const *, DruckerPragerCap_InitializeFunctions, nullptr);


MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_nPostProcessings, 0u);

MFRONT_EXPORT_SYMBOL(const char * const *, DruckerPragerCap_PostProcessings, nullptr);


MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_ComputesInternalEnergy, 0u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_ComputesDissipatedEnergy, 0u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_UsableInPurelyImplicitResolution, 1u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_nMaterialProperties, 0u);

MFRONT_EXPORT_SYMBOL(const char * const *, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_MaterialProperties, nullptr);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_nInternalStateVariables, 4u);
MFRONT_EXPORT_ARRAY_OF_SYMBOLS(const char *, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_InternalStateVariables, 4, MFRONT_EXPORT_ARRAY_ARGUMENTS("ElasticStrain",
"EquivalentPlasticStrain[0]","EquivalentPlasticStrain[1]","AxialStrain"));

MFRONT_EXPORT_ARRAY_OF_SYMBOLS(int, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_InternalStateVariablesTypes, 4, MFRONT_EXPORT_ARRAY_ARGUMENTS(1,0,0,0));

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_nExternalStateVariables, 1u);
MFRONT_EXPORT_ARRAY_OF_SYMBOLS(const char *, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_ExternalStateVariables, 1, MFRONT_EXPORT_ARRAY_ARGUMENTS("AxialStress"));

MFRONT_EXPORT_ARRAY_OF_SYMBOLS(int, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_ExternalStateVariablesTypes, 1, MFRONT_EXPORT_ARRAY_ARGUMENTS(0));

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_nParameters, 13u);

MFRONT_EXPORT_ARRAY_OF_SYMBOLS(const char *, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_Parameters, 13, MFRONT_EXPORT_ARRAY_ARGUMENTS("RelativeValueForTheEquivalentStressLowerBoundDefinition",
"epsilon","theta","YoungModulus","PoissonRatio","d",
"beta","pa","pb","minimal_time_step_scaling_factor","maximal_time_step_scaling_factor",
"numerical_jacobian_epsilon","iterMax"));

MFRONT_EXPORT_ARRAY_OF_SYMBOLS(int, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_ParametersTypes, 13, MFRONT_EXPORT_ARRAY_ARGUMENTS(0,0,0,0,0,0,0,0,0,0,0,0,2));

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_RelativeValueForTheEquivalentStressLowerBoundDefinition_ParameterDefaultValue, 1e-12);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_epsilon_ParameterDefaultValue, 1e-16);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_theta_ParameterDefaultValue, 1);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_YoungModulus_ParameterDefaultValue, 150000000000);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_PoissonRatio_ParameterDefaultValue, 0.3);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_d_ParameterDefaultValue, 150000000);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_beta_ParameterDefaultValue, 0.6);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_pa_ParameterDefaultValue, -10000000);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_pb_ParameterDefaultValue, -75000000);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_minimal_time_step_scaling_factor_ParameterDefaultValue, 0.1);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_maximal_time_step_scaling_factor_ParameterDefaultValue, 1.7976931348623e+308);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_numerical_jacobian_epsilon_ParameterDefaultValue, 1e-17);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_iterMax_ParameterDefaultValue, 100);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_requiresStiffnessTensor, 0u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_requiresThermalExpansionCoefficientTensor, 0u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_nInitializeFunctions, 0u);

MFRONT_EXPORT_SYMBOL(const char * const *, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_InitializeFunctions, nullptr);


MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_nPostProcessings, 0u);

MFRONT_EXPORT_SYMBOL(const char * const *, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_PostProcessings, nullptr);


MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_ComputesInternalEnergy, 0u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_ComputesDissipatedEnergy, 0u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_PlaneStress_UsableInPurelyImplicitResolution, 1u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_PlaneStress_nMaterialProperties, 0u);

MFRONT_EXPORT_SYMBOL(const char * const *, DruckerPragerCap_PlaneStress_MaterialProperties, nullptr);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_PlaneStress_nInternalStateVariables, 4u);
MFRONT_EXPORT_ARRAY_OF_SYMBOLS(const char *, DruckerPragerCap_PlaneStress_InternalStateVariables, 4, MFRONT_EXPORT_ARRAY_ARGUMENTS("ElasticStrain",
"EquivalentPlasticStrain[0]","EquivalentPlasticStrain[1]","AxialStrain"));

MFRONT_EXPORT_ARRAY_OF_SYMBOLS(int, DruckerPragerCap_PlaneStress_InternalStateVariablesTypes, 4, MFRONT_EXPORT_ARRAY_ARGUMENTS(1,0,0,0));

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_PlaneStress_nExternalStateVariables, 0u);
MFRONT_EXPORT_SYMBOL(const char * const *, DruckerPragerCap_PlaneStress_ExternalStateVariables, nullptr);

MFRONT_EXPORT_SYMBOL(const int *, DruckerPragerCap_PlaneStress_ExternalStateVariablesTypes, nullptr);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_PlaneStress_nParameters, 13u);

MFRONT_EXPORT_ARRAY_OF_SYMBOLS(const char *, DruckerPragerCap_PlaneStress_Parameters, 13, MFRONT_EXPORT_ARRAY_ARGUMENTS("RelativeValueForTheEquivalentStressLowerBoundDefinition",
"epsilon","theta","YoungModulus","PoissonRatio","d",
"beta","pa","pb","minimal_time_step_scaling_factor","maximal_time_step_scaling_factor",
"numerical_jacobian_epsilon","iterMax"));

MFRONT_EXPORT_ARRAY_OF_SYMBOLS(int, DruckerPragerCap_PlaneStress_ParametersTypes, 13, MFRONT_EXPORT_ARRAY_ARGUMENTS(0,0,0,0,0,0,0,0,0,0,0,0,2));

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_PlaneStress_RelativeValueForTheEquivalentStressLowerBoundDefinition_ParameterDefaultValue, 1e-12);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_PlaneStress_epsilon_ParameterDefaultValue, 1e-16);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_PlaneStress_theta_ParameterDefaultValue, 1);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_PlaneStress_YoungModulus_ParameterDefaultValue, 150000000000);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_PlaneStress_PoissonRatio_ParameterDefaultValue, 0.3);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_PlaneStress_d_ParameterDefaultValue, 150000000);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_PlaneStress_beta_ParameterDefaultValue, 0.6);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_PlaneStress_pa_ParameterDefaultValue, -10000000);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_PlaneStress_pb_ParameterDefaultValue, -75000000);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_PlaneStress_minimal_time_step_scaling_factor_ParameterDefaultValue, 0.1);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_PlaneStress_maximal_time_step_scaling_factor_ParameterDefaultValue, 1.7976931348623e+308);

MFRONT_EXPORT_SYMBOL(double, DruckerPragerCap_PlaneStress_numerical_jacobian_epsilon_ParameterDefaultValue, 1e-17);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_PlaneStress_iterMax_ParameterDefaultValue, 100);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_PlaneStress_requiresStiffnessTensor, 0u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_PlaneStress_requiresThermalExpansionCoefficientTensor, 0u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_PlaneStress_nInitializeFunctions, 0u);

MFRONT_EXPORT_SYMBOL(const char * const *, DruckerPragerCap_PlaneStress_InitializeFunctions, nullptr);


MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_PlaneStress_nPostProcessings, 0u);

MFRONT_EXPORT_SYMBOL(const char * const *, DruckerPragerCap_PlaneStress_PostProcessings, nullptr);


MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_PlaneStress_ComputesInternalEnergy, 0u);

MFRONT_EXPORT_SYMBOL(unsigned short, DruckerPragerCap_PlaneStress_ComputesDissipatedEnergy, 0u);

MFRONT_SHAREDOBJ void
DruckerPragerCap_setOutOfBoundsPolicy(const int p){
if(p==0){
DruckerPragerCap_getOutOfBoundsPolicy() = tfel::material::None;
} else if(p==1){
DruckerPragerCap_getOutOfBoundsPolicy() = tfel::material::Warning;
} else if(p==2){
DruckerPragerCap_getOutOfBoundsPolicy() = tfel::material::Strict;
} else {
std::cerr << "DruckerPragerCap_setOutOfBoundsPolicy: invalid argument\n";
}
}

MFRONT_SHAREDOBJ int
DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_setParameter(const char *const key,const double value){
using tfel::material::DruckerPragerCapAxisymmetricalGeneralisedPlaneStressParametersInitializer;
auto& i = DruckerPragerCapAxisymmetricalGeneralisedPlaneStressParametersInitializer::get();
try{
i.set(key,value);
} catch(std::runtime_error& e){
std::cerr << e.what() << std::endl;
return 0;
}
return 1;
}

MFRONT_SHAREDOBJ int
DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress_setUnsignedShortParameter(const char *const key,const unsigned short value){
using tfel::material::DruckerPragerCapAxisymmetricalGeneralisedPlaneStressParametersInitializer;
auto& i = DruckerPragerCapAxisymmetricalGeneralisedPlaneStressParametersInitializer::get();
try{
i.set(key,value);
} catch(std::runtime_error& e){
std::cerr << e.what() << std::endl;
return 0;
}
return 1;
}

MFRONT_SHAREDOBJ int
DruckerPragerCap_PlaneStress_setParameter(const char *const key,const double value){
using tfel::material::DruckerPragerCapPlaneStressParametersInitializer;
auto& i = DruckerPragerCapPlaneStressParametersInitializer::get();
try{
i.set(key,value);
} catch(std::runtime_error& e){
std::cerr << e.what() << std::endl;
return 0;
}
return 1;
}

MFRONT_SHAREDOBJ int
DruckerPragerCap_PlaneStress_setUnsignedShortParameter(const char *const key,const unsigned short value){
using tfel::material::DruckerPragerCapPlaneStressParametersInitializer;
auto& i = DruckerPragerCapPlaneStressParametersInitializer::get();
try{
i.set(key,value);
} catch(std::runtime_error& e){
std::cerr << e.what() << std::endl;
return 0;
}
return 1;
}

MFRONT_SHAREDOBJ int
DruckerPragerCap_setParameter(const char *const key,const double value){
using tfel::material::DruckerPragerCapParametersInitializer;
auto& i = DruckerPragerCapParametersInitializer::get();
try{
i.set(key,value);
} catch(std::runtime_error& e){
std::cerr << e.what() << std::endl;
return 0;
}
return 1;
}

MFRONT_SHAREDOBJ int
DruckerPragerCap_setUnsignedShortParameter(const char *const key,const unsigned short value){
using tfel::material::DruckerPragerCapParametersInitializer;
auto& i = DruckerPragerCapParametersInitializer::get();
try{
i.set(key,value);
} catch(std::runtime_error& e){
std::cerr << e.what() << std::endl;
return 0;
}
return 1;
}

MFRONT_SHAREDOBJ int DruckerPragerCap_AxisymmetricalGeneralisedPlaneStrain(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr auto h = ModellingHypothesis::AXISYMMETRICALGENERALISEDPLANESTRAIN;
using Behaviour = DruckerPragerCap<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d, Behaviour::STANDARDTANGENTOPERATOR, DruckerPragerCap_getOutOfBoundsPolicy());
return r;
} // end of DruckerPragerCap_AxisymmetricalGeneralisedPlaneStrain

MFRONT_SHAREDOBJ int DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr auto h = ModellingHypothesis::AXISYMMETRICALGENERALISEDPLANESTRESS;
using Behaviour = DruckerPragerCap<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d, Behaviour::STANDARDTANGENTOPERATOR, DruckerPragerCap_getOutOfBoundsPolicy());
return r;
} // end of DruckerPragerCap_AxisymmetricalGeneralisedPlaneStress

MFRONT_SHAREDOBJ int DruckerPragerCap_Axisymmetrical(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr auto h = ModellingHypothesis::AXISYMMETRICAL;
using Behaviour = DruckerPragerCap<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d, Behaviour::STANDARDTANGENTOPERATOR, DruckerPragerCap_getOutOfBoundsPolicy());
return r;
} // end of DruckerPragerCap_Axisymmetrical

MFRONT_SHAREDOBJ int DruckerPragerCap_PlaneStress(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr auto h = ModellingHypothesis::PLANESTRESS;
using Behaviour = DruckerPragerCap<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d, Behaviour::STANDARDTANGENTOPERATOR, DruckerPragerCap_getOutOfBoundsPolicy());
return r;
} // end of DruckerPragerCap_PlaneStress

MFRONT_SHAREDOBJ int DruckerPragerCap_PlaneStrain(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr auto h = ModellingHypothesis::PLANESTRAIN;
using Behaviour = DruckerPragerCap<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d, Behaviour::STANDARDTANGENTOPERATOR, DruckerPragerCap_getOutOfBoundsPolicy());
return r;
} // end of DruckerPragerCap_PlaneStrain

MFRONT_SHAREDOBJ int DruckerPragerCap_GeneralisedPlaneStrain(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr auto h = ModellingHypothesis::GENERALISEDPLANESTRAIN;
using Behaviour = DruckerPragerCap<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d, Behaviour::STANDARDTANGENTOPERATOR, DruckerPragerCap_getOutOfBoundsPolicy());
return r;
} // end of DruckerPragerCap_GeneralisedPlaneStrain

MFRONT_SHAREDOBJ int DruckerPragerCap_Tridimensional(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr auto h = ModellingHypothesis::TRIDIMENSIONAL;
using Behaviour = DruckerPragerCap<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d, Behaviour::STANDARDTANGENTOPERATOR, DruckerPragerCap_getOutOfBoundsPolicy());
return r;
} // end of DruckerPragerCap_Tridimensional

#ifdef __cplusplus
}
#endif /* __cplusplus */

