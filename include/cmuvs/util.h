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

/*
 * @brief Evaluates the SE(3) transformation for given parameters. Computes
 * the transformation:
 *  \f[ H = A \cdot \exp( p_0 \cdot [\mathbf{v}_0]_{\wedge} ) \cdot ...
 *  \cdot \exp(p_{n-1} \cdot [\mathbf{v}_{n-1}]_{\wedge}) \f]
 * where `A` is the base transformation, and each `exp(param_i * [vec_i]_w)` represents an 
 * exponential map of an `se(3)` Lie algebra element.
 *
 * @param params A vector of parameters \f$ \mathbf{p} \in \mathbb{R}^{n} \f$
 *               used in the exponentials.
 * @return The evaluated \f$ 4 \times 4 \f$ homogeneous transformation matrix.
 */
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
    Eigen::Matrix<T, 4, 4> res = Eigen::Matrix<T, 4, 4>::Identity();
    Eigen::Vector<T, 3> v = eta.head(3);
    Eigen::Vector<T, 3> w = eta.tail(3);

    //std::cout << "v:\n" << v << "\n";
    //std::cout << "w:\n" << w << "\n";

    T theta = (T)w.norm();
    Eigen::Vector<T, 3> unitVec = w.normalized();

    if (theta < 1e-8) {  // Pure translation, no rotation
        res.template block<3,1>(0,3) = v;
        return res;
    }

    Eigen::Matrix<T, 3, 3> K = skew(unitVec);
    Eigen::Matrix<T, 3, 3> R = Eigen::Matrix<T, 3, 3>::Identity() + sin(theta)*K + (1.0 - cos(theta))*K*K;
    Eigen::Matrix<T, 3, 3> V = Eigen::Matrix<T, 3, 3>::Identity() + ((1-cos(theta))/theta)*K + (1-sin(theta)/theta)*K*K;

    //std::cout << "R: " << R << "\n";
    //std::cout << "Rv: " << R*v << "\n";
    res.template block<3,3>(0,0) = R;
    res.template block<3,1>(0,3) = V*v;
    return res;
}

template <typename Derived>
Eigen::Matrix<double,4,4> se3Exp(const Eigen::MatrixBase<Derived>& twist) {
    Eigen::Matrix<double,6,1> evaluatedTwist = twist.eval();  // Ensure it's evaluated
    return se3Exp(evaluatedTwist);  // Call actual implementation
}

// [adjointMatrix(H) * v]_{wedge} = H [v]_{wedge} * H^{-1}
Eigen::Matrix<double, 6, 6> adjointMatrix(const Eigen::Isometry3d& T);

// adjointMatrix(exp(v))
Eigen::Matrix<double, 6, 6> adjointMatrix(const Eigen::Vector<double,6>& v);

// [adjointInvMatrix(H) * v]_{wedge} = H^{-1} [v]_{wedge} * H
Eigen::Matrix<double, 6, 6> adjointInvMatrix(const Eigen::Isometry3d& T);

// adjointInvMatrix(exp(v))
Eigen::Matrix<double, 6, 6> adjointInvMatrix(const Eigen::Vector<double,6>& v);


