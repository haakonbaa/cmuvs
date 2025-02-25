#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>
#include <cassert>

#include <cmuvs/util.h>

// SE3Term represents a transformation of
//      exp( param_{paramIndex}  * [twist]_{\wedge} )
struct SE3Term {
    Eigen::Vector<double, 6> twist; // Element of se(3)
    unsigned int paramIndex;
};

class SE3 {
// Represents the transform:
//   A * exp( param_0 * [vec_0]_w ) * ... * exp(param_{n-1} * [vec_{n-1}]_w)
private:
    Eigen::Isometry3d A;
    std::vector<SE3Term> exponentials;
    
public:
    ~SE3() = default;
    SE3();
    explicit SE3(const Eigen::Isometry3d& baseTransform); // Identity
    Eigen::Isometry3d getBaseTransform() const; 
    void addExponential(const Eigen::Matrix<double, 6, 1>& twist, unsigned int paramIndex);

    /**
    * @brief Evaluates the SE(3) transformation for given parameters. Computes
    * the transformation:
    * \f[
    * H = A \cdot \exp( p_0 \cdot [\mathbf{v}_0]_{\wedge} ) \cdot ... \cdot \exp(p_{n-1} \cdot [\mathbf{v}_{n-1}]_{\wedge})
    * \f]
    * where `A` is the base transformation, and each `exp(param_i * [vec_i]_w)` represents an 
    * exponential map of an `se(3)` Lie algebra element.
    *
    * @param params A vector of parameters \f$ \mathbf{p} \in \mathbb{R}^{n} \f$
    *               used in the exponentials.
    * @return The evaluated \f$ 4 \times 4 \f$ homogeneous transformation matrix.
    */
    Eigen::Matrix4d eval(const Eigen::VectorXd& params) const;
    Eigen::Vector3d evalVec(const Eigen::VectorXd& params, const Eigen::Vector3d x) const;
    Eigen::Vector<double,6> twistBody(const Eigen::VectorXd& params, const Eigen::VectorXd diffParams) const;
    Eigen::Matrix<double,6,Eigen::Dynamic> jacobian(const Eigen::VectorXd& params, Eigen::Matrix<double,6,Eigen::Dynamic> Jold = Eigen::Matrix<double,6,0>::Zero());

    static SE3 rotateX(double angle);
    static SE3 rotateY(double angle);
    static SE3 rotateZ(double angle);
    static SE3 translate(const Eigen::Vector3d& v);
    static SE3 transformParam(const Eigen::Vector<double, 6>& twist, unsigned int paramIndex);
    static SE3 rotateXParam(unsigned int paramIndex);
    static SE3 rotateYParam(unsigned int paramIndex);
    static SE3 rotateZParam(unsigned int paramIndex);
    static SE3 translateParam(const Eigen::Vector3d& v, unsigned int paramIndex);

    friend std::ostream& operator<<(std::ostream& os, const SE3& H);
    SE3 operator*(const SE3& rhs);
};
