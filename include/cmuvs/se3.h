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
    Eigen::Matrix4d eval(const Eigen::VectorXd& params) const;

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
