#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "UKF/Types.h"
#include "UKF/Integrator.h"
#include "UKF/StateVector.h"
#include "UKF/MeasurementVector.h"
#include "UKF/Core.h"

constexpr double DELTA_T = 0.01;

enum MyStateFields {
    Position1,
    Velocity1,
    Position2,
    Velocity2,
    Position3,
    Velocity3,
};

using MyStateVector = UKF::StateVector<
    UKF::Field<Position1, UKF::Vector<3>>,
    UKF::Field<Velocity1, UKF::Vector<3>>,
    UKF::Field<Position2, UKF::Vector<3>>,
    UKF::Field<Velocity2, UKF::Vector<3>>,
    UKF::Field<Position3, UKF::Vector<3>>,
    UKF::Field<Velocity3, UKF::Vector<3>>
>;

//namespace Parameters {
//template <> constexpr real_t AlphaSquared<MyStateVector> = 1e-6;
//}

namespace UKF {
/*
State vector process model. One version takes body frame kinematic
acceleration and angular acceleration as inputs, the other doesn't (assumes
zero accelerations).
*/
template <> template <>
MyStateVector MyStateVector::derivative<>() const {
    MyStateVector temp;
    temp.set_field<Position1>(get_field<Velocity1>());
    temp.set_field<Position2>(get_field<Velocity2>());
    temp.set_field<Position3>(get_field<Velocity3>());
    temp.set_field<Velocity1>(Vector<3>::Zero());
    temp.set_field<Velocity2>(Vector<3>::Zero());
    temp.set_field<Velocity3>(Vector<3>::Zero());
    return temp;
}

}

/* Set up measurement vector class. */
enum MyMeasurementFields {
    GPS_Position1,
    GPS_Velocity1,
    GPS_Position2,
    GPS_Velocity2,
    GPS_Position3,
    GPS_Velocity3
};

using MyMeasurementVector = UKF::DynamicMeasurementVector<
    UKF::Field<GPS_Position1, UKF::Vector<3>>,
    UKF::Field<GPS_Velocity1, UKF::Vector<3>>,
    UKF::Field<GPS_Position2, UKF::Vector<3>>,
    UKF::Field<GPS_Velocity2, UKF::Vector<3>>,
    UKF::Field<GPS_Position3, UKF::Vector<3>>,
    UKF::Field<GPS_Velocity3, UKF::Vector<3>>
>;

using MyCore = UKF::Core<
    MyStateVector,
    MyMeasurementVector,
    UKF::IntegratorEuler
>;

template <> template <>
UKF::Vector<3> MyMeasurementVector::expected_measurement
<MyStateVector, GPS_Position1>(const MyStateVector& state) {
    return state.get_field<Position1>();
}

template <> template <>
UKF::Vector<3> MyMeasurementVector::expected_measurement
<MyStateVector, GPS_Velocity1>(const MyStateVector& state) {
    return state.get_field<Velocity1>();
}

template <> template <>
UKF::Vector<3> MyMeasurementVector::expected_measurement
<MyStateVector, GPS_Position2>(const MyStateVector& state) {
    return state.get_field<Position2>();
}

template <> template <>
UKF::Vector<3> MyMeasurementVector::expected_measurement
<MyStateVector, GPS_Velocity2>(const MyStateVector& state) {
    return state.get_field<Velocity2>();
}

template <> template <>
UKF::Vector<3> MyMeasurementVector::expected_measurement
<MyStateVector, GPS_Position3>(const MyStateVector& state) {
    return state.get_field<Position3>();
}

template <> template <>
UKF::Vector<3> MyMeasurementVector::expected_measurement
<MyStateVector, GPS_Velocity3>(const MyStateVector& state) {
    return state.get_field<Velocity3>();
}

MyCore create_initialised_test_filter() {
    MyCore test_filter;
    test_filter.state.set_field<Position1>(UKF::Vector<3>(0, 0, 0));
    test_filter.state.set_field<Velocity1>(UKF::Vector<3>(0, 0, 0));
    test_filter.state.set_field<Position2>(UKF::Vector<3>(1, 0, 0));
    test_filter.state.set_field<Velocity2>(UKF::Vector<3>(1, 0, 0));
    test_filter.state.set_field<Position3>(UKF::Vector<3>(2, 0, 0));
    test_filter.state.set_field<Velocity3>(UKF::Vector<3>(2, 0, 0));
    test_filter.covariance = MyStateVector::CovarianceMatrix::Identity() * 100;
    test_filter.measurement_covariance = UKF::Vector<18>::Ones(); // Matrix diagonal

    real_t a, b;
    real_t dt = DELTA_T;
    a = std::sqrt(0.1*dt*dt);
    b = std::sqrt(0.1*dt);
    test_filter.process_noise_covariance = MyStateVector::CovarianceMatrix::Identity() * a;

    static_assert(MyStateVector::size() == 18, "");

    return test_filter;
}

void ukfPredict(benchmark::State& state) {
    MyCore test_filter = create_initialised_test_filter();

    while(state.KeepRunning()) {
        test_filter.a_priori_step(DELTA_T);
    }
}
BENCHMARK(ukfPredict);


void ukfPredictCorrect(benchmark::State& state) {
    MyCore test_filter = create_initialised_test_filter();
    using MeasurementVector1 = UKF::DynamicMeasurementVector<
      UKF::Field<Position1, UKF::Vector<3>>,
      UKF::Field<Velocity1, UKF::Vector<3>>
    >;
    using MeasurementVector2 = UKF::DynamicMeasurementVector<
      UKF::Field<Position2, UKF::Vector<3>>,
      UKF::Field<Velocity2, UKF::Vector<3>>
    >;
    using MeasurementVector3 = UKF::DynamicMeasurementVector<
      UKF::Field<Position3, UKF::Vector<3>>,
      UKF::Field<Velocity3, UKF::Vector<3>>
    >;

    UKF::Vector<3> truePos1 = UKF::Vector<3>::Zero();
    const UKF::Vector<3> trueVelocity1 = UKF::Vector<3>(10.0, 1.0, -5.0);
    const UKF::Vector<3> trueScaledVelocity1 = trueVelocity1 * DELTA_T;
    UKF::Vector<3> truePos2 = UKF::Vector<3>::Zero();
    const UKF::Vector<3> trueVelocity2 = UKF::Vector<3>(0.1, 0.1, 0.1);
    const UKF::Vector<3> trueScaledVelocity2 = trueVelocity2 * DELTA_T;
    UKF::Vector<3> truePos3 = UKF::Vector<3>::Zero();
    const UKF::Vector<3> trueVelocity3 = UKF::Vector<3>(0.5, 1.0, 2.0);
    const UKF::Vector<3> trueScaledVelocity3 = trueVelocity3 * DELTA_T;
    size_t i = 0;
    while(state.KeepRunning()) {
      test_filter.a_priori_step(DELTA_T);
      // 1
      {
        MeasurementVector1 meas;
        meas.set_field<Position1>(truePos1 + i * trueScaledVelocity1);
        meas.set_field<Velocity1>(trueVelocity1);
        test_filter.innovation_step(meas);
        test_filter.a_posteriori_step();
      }
      // 2
      {
        MeasurementVector2 meas;
        meas.set_field<Position2>(truePos2 + i * trueScaledVelocity2);
        meas.set_field<Velocity2>(trueVelocity2);
        test_filter.innovation_step(meas);
        test_filter.a_posteriori_step();
      }
      // 3
      {
        MeasurementVector3 meas;
        meas.set_field<Position3>(truePos3 + i * trueScaledVelocity3);
        meas.set_field<Velocity3>(trueVelocity3);
        test_filter.innovation_step(meas);
        test_filter.a_posteriori_step();
      }
      ++i;
    }
}
BENCHMARK(ukfPredictCorrect);