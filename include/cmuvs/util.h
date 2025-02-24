#pragma once
#include <Eigen/Dense>
#include <type_traits>

#include <iostream> // TODO: remove

template<typename Scalar_>
Eigen::Matrix<Scalar_, 3, 3> skew(Eigen::Matrix<Scalar_,3,1> v) {
    Eigen::Matrix<Scalar_,3,3> res = Eigen::Matrix<Scalar_,3,3>::Zero();
    res(0,1) = -v(2); res(1,0) = v(2);
    res(0,2) = v(1); res(2,0) = -v(1);
    res(1,2) = -v(0); res(2,1) = v(0);
    return res;
}

template<typename Scalar_>
Eigen::Matrix<Scalar_, 4, 4> wedge(const Eigen::Matrix<Scalar_,6,1>& v) {
    Eigen::Matrix<Scalar_, 4, 4> res = Eigen::Matrix<Scalar_, 4, 4>::Zero();
    res.template block<3,3>(0,0) = skew(v.template block<3,1>(3,0).eval());
    res.template block<3,1>(0,3) = v.template block<3,1>(0,0);
    return res;
}

Eigen::Matrix4d wedge(const Eigen::Ref<const Eigen::Matrix<double,6,1>>& v);

template<typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
Eigen::Matrix<T, 3, 3> so3Exp(const Eigen::Vector<T, 3>& v)
{
    Eigen::Vector<T, 3> unitVec = Eigen::Vector<T, 3>::Zero();
    T angle = (T)0.0;

    if (v.norm() >= (T)1e-6) {
        unitVec = v.normalized();
        angle = v.norm();
    }
    Eigen::Matrix<T, 3, 3> K = skew(unitVec);
    return Eigen::Matrix<T, 3, 3>::Identity() + sin(angle)*K + (1.0-cos(angle))*K*K;
}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
Eigen::Matrix<T, 4, 4> se3Exp(const Eigen::Vector<T, 6>& eta)
{
    Eigen::Vector<T, 3> v = eta.head(3);
    Eigen::Vector<T, 3> w = eta.tail(3);

    //std::cout << "v: " << v << "\n"; 
    //std::cout << "w: " << w << "\n"; 

    Eigen::Vector<T, 3> unitVec = w.normalized();
    T theta = (T)w.norm();

    Eigen::Matrix<T, 3, 3> K = skew(unitVec);
    Eigen::Matrix<T, 3, 3> R = Eigen::Matrix<T, 3, 3>::Identity() + sin(theta)*K + (1.0 - cos(theta))*K*K;

    Eigen::Matrix<T, 4, 4> res = Eigen::Matrix<T, 4, 4>::Identity();
    //std::cout << "R: " << R << "\n";
    //std::cout << "Rv: " << R*v << "\n";
    res.template block<3,3>(0,0) = R;
    res.template block<3,1>(0,3) = R*v;
    return res;
}

template <typename Derived>
Eigen::Matrix<double,4,4> se3Exp(const Eigen::MatrixBase<Derived>& twist) {
    Eigen::Matrix<double,6,1> evaluatedTwist = twist.eval();  // Ensure it's evaluated
    return se3Exp(evaluatedTwist);  // Call actual implementation
}

// [adjointMatrix(H) * v]_{wedge} = H [v]_{wedge} * H^{-1}
Eigen::Matrix<double, 6, 6> adjointMatrix(const Eigen::Isometry3d& T);

// [adjointInvMatrix(H) * v]_{wedge} = H^{-1} [v]_{wedge} * H
Eigen::Matrix<double, 6, 6> adjointInvMatrix(const Eigen::Isometry3d& T);


