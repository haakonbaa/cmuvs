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

// Evaluates the SE(3) object, 'H', multiplied by a homogeneous vector x
//       H * [x^T 1]^T
// This is more efficient than first calling eval then multiplying
Eigen::Vector3d SE3::evalVec(const Eigen::VectorXd& params, const Eigen::Vector3d x) const {
    Eigen::Vector4d res;
    res << x(0), x(1), x(2), 1.0;
    for (std::vector<SE3Term>::const_reverse_iterator riter = this->exponentials.rbegin();
            riter != this->exponentials.rend(); riter++)
    {
        assert(params.rows() > riter->paramIndex && "Attempt to eval SE(3) object with to few parameters!");
        res = se3Exp(params(riter->paramIndex) * riter->twist) * res;
    }
    res = this->A.matrix() * res;
    return res.block<3,1>(0,0);
}

Eigen::Vector<double,6> SE3::twistBody(const Eigen::VectorXd& params, const Eigen::VectorXd diffParams) const
{
    std::cout << "\x1b[31mWARNING\x1b[0m: SE3::twistBody not verified!\n";
    Eigen::Vector<double,6> twist = Eigen::Vector<double,6>::Zero();
    Eigen::Matrix<double,6,6> preMul = Eigen::Matrix<double,6,6>::Identity();

    for (std::vector<SE3Term>::const_reverse_iterator riter = this->exponentials.rbegin();
            riter != this->exponentials.rend(); riter++)
    {
        assert(params.rows() > riter->paramIndex && "Attempt to eval SE(3) object with to few parameters!");
        assert(diffParams.rows() > riter->paramIndex && "Attempt to eval SE(3) object with to few parameters!");

        twist = preMul * diffParams(riter->paramIndex) * riter->twist;
        preMul = preMul * adjointInvMatrix(params(riter->paramIndex)*riter->twist);
    }
    return twist;
}

// creates a 6xcol Matrix with the vector a at index n
[[maybe_unused]] static Eigen::Matrix<double,6,Eigen::Dynamic> insertMatrix(int col, int n, const Eigen::Vector<double,6>& a) {
    assert(col > 0 && "Cannot make 0 cols matrix!");
    assert(n < col && "Matrix to small to insert at index!");

    Eigen::Matrix<double,6,Eigen::Dynamic> M = Eigen::MatrixXd::Zero(6,col);
    M.block<6,1>(0,n) = a;
    return M;
}

// Returns a vector J such that the body twist eta = J(params) * params_dot
Eigen::Matrix<double,6,Eigen::Dynamic> SE3::jacobian(const Eigen::VectorXd& params, [[maybe_unused]] Eigen::Matrix<double,6,Eigen::Dynamic> Jold) {
    int cols = params.size();

    if (Jold.cols() == 0) {
        Jold = Eigen::MatrixXd::Zero(6,cols);
    }
    assert(Jold.rows() == 6 && Jold.cols() == cols && "Old jacobian does not fit size determined by params");

    Eigen::Matrix<double,6,Eigen::Dynamic> J = Eigen::MatrixXd::Zero(6,cols);
    Eigen::Matrix<double,6,6> preMul = Eigen::Matrix<double,6,6>::Identity();

    for (std::vector<SE3Term>::const_reverse_iterator riter = this->exponentials.rbegin();
            riter != this->exponentials.rend(); riter++)
    {
        assert(cols > (int)riter->paramIndex && "Attempt to eval SE(3) object with to few parameters!");

        J += preMul * insertMatrix(cols, riter->paramIndex, riter->twist);
        preMul = preMul * adjointInvMatrix(params(riter->paramIndex)*riter->twist);
    }
    preMul = preMul * adjointInvMatrix(this->A);
    J += preMul * Jold;

    // TODO: Implement tests

    return J;
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
