#include <cmuvs/se3.h>

SE3::SE3() : A(Eigen::Isometry3d::Identity()), exponentials() {}
SE3::SE3(const Eigen::Isometry3d& baseTransform) : A(baseTransform), exponentials() {}

void SE3::addExponential(const Eigen::Matrix<double, 6, 1>& twist, unsigned int paramIndex) {
    this->exponentials.push_back({twist, paramIndex});
}

Eigen::Isometry3d SE3::getBaseTransform() const {
    return this->A;
}

SE3 SE3::operator*(const SE3& rhs) {
    SE3 res = SE3(this->A * rhs.A);
    res.exponentials.reserve(this->exponentials.size() + rhs.exponentials.size());
    for (const SE3Term& term : this->exponentials) {
        res.addExponential((adjointInvMatrix(rhs.A)*term.twist).eval(), term.paramIndex);
    }
    for (const SE3Term& term : rhs.exponentials) {
        res.addExponential(term.twist, term.paramIndex);
    }
    return res;
}

Eigen::Matrix4d SE3::eval(const Eigen::VectorXd& params) const {
    Eigen::Matrix4d res = (this->A).matrix();
    for (const SE3Term& term : this->exponentials) {
        assert(params.rows() > term.paramIndex && "Attempt to eval SE(3) object with to few parameters!");
        res = res * se3Exp((params(term.paramIndex) * term.twist).eval());
    }
    return res;
}

std::ostream& operator<<(std::ostream& os, const SE3& H) {
    os << H.getBaseTransform().matrix();
    for (const SE3Term& term : H.exponentials) {
        os << " exp(v_" << term.paramIndex << "[" << term.twist.transpose() << "])";
    }
    return os;
}

// Static Member functions ----------------------------------------------------

SE3 SE3::rotateX(double angle) {
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.rotate(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()));
    return SE3(transform);
}

SE3 SE3::rotateY(double angle) {
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.rotate(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()));
    return SE3(transform);
}

SE3 SE3::rotateZ(double angle) {
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.rotate(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()));
    return SE3(transform);
}

SE3 SE3::translate(const Eigen::Vector3d& v) {
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.pretranslate(v);
    return SE3(transform);
}

SE3 SE3::transformParam(const Eigen::Vector<double, 6>& twist, unsigned int paramIndex) {
    SE3 transform = SE3(); // Identity element
    transform.addExponential(twist, paramIndex);
    return transform;
}

SE3 SE3::rotateXParam(unsigned int paramIndex) {
    Eigen::Vector<double, 6> twist;
    twist << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
    return SE3::transformParam(twist, paramIndex);
}

SE3 SE3::rotateYParam(unsigned int paramIndex) {
    Eigen::Vector<double, 6> twist;
    twist << 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
    return SE3::transformParam(twist, paramIndex);
}

SE3 SE3::rotateZParam(unsigned int paramIndex) {
    Eigen::Vector<double, 6> twist;
    twist << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    return SE3::transformParam(twist, paramIndex);
}

SE3 SE3::translateParam(const Eigen::Vector3d& v, unsigned int paramIndex) {
    Eigen::Vector<double, 6> twist;
    twist << v(0), v(1), v(2), 0.0, 0.0, 0.0;
    return SE3::transformParam(twist, paramIndex);
}
