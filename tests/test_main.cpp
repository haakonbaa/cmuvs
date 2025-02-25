#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

#include "../include/cmuvs/util.h"
#include "../include/cmuvs/se3.h"

// Naive matrix exponentiation for testing
template<int Size_>
static Eigen::Matrix<double, Size_, Size_> exp(const Eigen::Matrix<double, Size_, Size_>& A) {
    Eigen::Matrix<double, Size_, Size_> An = Eigen::Matrix<double, Size_, Size_>::Identity();
    Eigen::Matrix<double, Size_, Size_> res = Eigen::Matrix<double, Size_, Size_>::Identity();
    double fact = 1.0;
    for (int i = 1; i < 100; i++) {
        fact *= i;
        An = An * A;
        res += An / fact;
    }
    return res;
}

static Eigen::Matrix<double,6,6> standardJacobian(Eigen::Vector<double,6> state) {
    double phi, theta, psi;
    phi = state(3);
    theta = state(4);
    psi = state(5);
    Eigen::Matrix<double,6,6> J;
    J <<                             cos(psi)*cos(theta),                               sin(psi)*cos(theta),         -sin(theta), 0, 0, 0,
         -sin(psi)*cos(phi)+cos(psi)*sin(theta)*sin(phi),  cos(psi)*cos(phi) + sin(phi)*sin(theta)*sin(psi), cos(theta)*sin(phi), 0, 0, 0,
        sin(psi)*sin(phi) + cos(psi)*cos(phi)*sin(theta), -cos(psi)*sin(phi) + sin(theta)*sin(psi)*cos(phi), cos(theta)*cos(phi), 0, 0, 0,
        0, 0, 0, 1,         0,         -sin(theta),
        0, 0, 0, 0,  cos(phi), sin(phi)*cos(theta),
        0, 0, 0, 0, -sin(phi), cos(phi)*cos(theta);
    return J;
}

TEST_CASE("Skew operator computation", "[skew]") {
    Eigen::Vector3i v;
    v << 1, 2, 3;
    Eigen::Matrix3i V;
    V <<  0, -3,  2,
          3,  0, -1,
         -2,  1,  0;
    Eigen::Vector3i a;
    a << 4, 4, 2;
    REQUIRE( skew(v) == V);
    REQUIRE( V * a == v.cross(a));
}

TEST_CASE("Wedge operator computation", "[wedge]") {
    Eigen::Matrix<int, 6, 1> v;
    v << 1,2,3,4,5,6;
    Eigen::Matrix4i V;
    V <<  0, -6,  5, 1,
          6,  0, -4, 2,
         -5,  4,  0, 3,
          0,  0,  0, 0;
    REQUIRE( wedge(v) == V);
}

TEST_CASE("SE(3) initializations", "[SE3]") {
    SE3 h = SE3();

    SE3 h2 = SE3::rotateX(M_PI/2.0);
    Eigen::Matrix4d h2e;
    h2e << 1.0, 0.0,  0.0, 0.0,
           0.0, 0.0, -1.0, 0.0,
           0.0, 1.0,  0.0, 0.0,
           0.0, 0.0,  0.0, 1.0;
    REQUIRE( (h2.getBaseTransform().matrix() - h2e).norm() < 1e-6);

    SE3 h3 = SE3::rotateY(M_PI);
    Eigen::Matrix4d h3e;
    h3e << -1.0, 0.0,  0.0, 0.0,
           0.0, 1.0, -0.0, 0.0,
           0.0, 0.0,  -1.0, 0.0,
           0.0, 0.0,  0.0, 1.0;
    REQUIRE( (h3.getBaseTransform().matrix() - h3e).norm() < 1e-6);

    SE3 h4 = SE3::rotateZ(M_PI/2.0);
    Eigen::Matrix4d h4e;
    h4e << 0.0, -1.0, 0.0, 0.0,
           1.0,  0.0, 0.0, 0.0,
           0.0,  0.0, 1.0, 0.0,
           0.0,  0.0, 0.0, 1.0;
    REQUIRE( (h4.getBaseTransform().matrix() - h4e).norm() < 1e-6);

    SE3 h5 = SE3::translate(Eigen::Vector3d(1.0, 2.0, 3.0));
    Eigen::Matrix4d h5e;
    h5e << 1.0, 0.0, 0.0, 1.0,
           0.0, 1.0, 0.0, 2.0,
           0.0, 0.0, 1.0, 3.0,
           0.0, 0.0, 0.0, 1.0;
    REQUIRE( (h5.getBaseTransform().matrix() - h5e).norm() < 1e-6);
}

TEST_CASE("so(3) exp", "[so3Exp]") {
    for (double theta : std::array<double,3>{0.0, 1.0, M_PI/2.0}) {
        Eigen::Vector3d x;
        x << theta, 0.0, 0.0;
        Eigen::Matrix3d X = so3Exp(x);
        Eigen::Matrix3d XE;
        XE << 1.0, 0.0, 0.0,
            0.0, cos(theta), -sin(theta),
            0.0, sin(theta), cos(theta);
        REQUIRE( (X - XE).norm() < 1e-6);

        Eigen::Vector3d y;
        y << 0.0, theta, 0.0;
        Eigen::Matrix3d Y = so3Exp(y);
        Eigen::Matrix3d YE;
        YE << cos(theta), 0.0, sin(theta),
              0.0, 1.0, 0.0,
              -sin(theta), 0.0, cos(theta);
        REQUIRE( (Y - YE).norm() < 1e-6);

        Eigen::Vector3d z;
        z << 0.0, 0.0, theta;
        Eigen::Matrix3d Z = so3Exp(z);
        Eigen::Matrix3d ZE;
        ZE << cos(theta), -sin(theta), 0.0,
           sin(theta), cos(theta), 0.0,
           0.0, 0.0, 1.0;
        REQUIRE( (Z - ZE).norm() < 1e-6);
    }

    Eigen::Vector3d w;
    w << 1.0, 2.0, 3.0;
    Eigen::Matrix3d W = skew(w);
    REQUIRE( (so3Exp(w) - exp(W)).norm() < 1e-6);
}

TEST_CASE("se(3) exp","[se3Exp]") {
    Eigen::Vector<double,6> v, v2;
    v << 1.0, 2.0, 3.0, 1.1, 2.2, 1.3;
    v2 << 1.0, 2.0, 3.0, 0.0, 0.0, 0.0;

    Eigen::Matrix<double,4,4> V = wedge(v);
    Eigen::Matrix<double,4,4> V2 = wedge(v2);

    REQUIRE( (exp(V) - se3Exp(v)).norm() < 1e-6);
    REQUIRE( (exp(V2) - se3Exp(v2)).norm() < 1e-6);
}


TEST_CASE("Adjoint operator", "[adjointMatrix]") {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(Eigen::AngleAxisd(1.523,Eigen::Vector3d(1.5,5.1,3.7).normalized()));
    T.translate(Eigen::Vector3d(-1.3,M_PI/1.4,sqrt(2.0)));

    Eigen::Vector<double,6> twist;
    twist << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

    Eigen::Matrix4d A1 = wedge(adjointMatrix(T) * twist);
    Eigen::Matrix4d A2 = T.matrix() * wedge(twist) * T.inverse().matrix();
    REQUIRE( (A1 - A2).norm() < 1e-6);

    Eigen::Matrix4d B1 = wedge(adjointInvMatrix(T) * twist);
    Eigen::Matrix4d B2 = T.inverse().matrix() * wedge(twist) * T.matrix();
    REQUIRE( (B1 - B2).norm() < 1e-6);
}

TEST_CASE("Adjoint operator se3", "[adjointMatrix]") {
    Eigen::Vector<double,6> eta, x;
    eta << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
    x << 1.2, 0.4, 0.3, 1.5, 0.6, 0.2;
    // [Ad(exp([eta])) x] = exp([eta]) * [x] (exp([eta]))^{-1}
    Eigen::Matrix4d A1 = wedge(adjointMatrix(eta) * x);
    Eigen::Matrix4d A2 = exp(wedge(eta)) * wedge(x) * exp(wedge(-eta));
}

TEST_CASE("ostream << SE3", "SE3::operator<<") {
    SE3 A = SE3();
    A.addExponential(Eigen::Vector<double,6>::Zero(),6);
    A.addExponential(Eigen::Vector<double,6>::Zero(),2);
    std::stringstream os;
    os << A;
    std::string output = os.str();
    REQUIRE(output == "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1 exp(v_6[0 0 0 0 0 0]) exp(v_2[0 0 0 0 0 0])");
}

TEST_CASE("SE3 evaluation", "SE3::eval()") {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(Eigen::AngleAxisd(2.3,Eigen::Vector3d::UnitX()));
    T.translate(Eigen::Vector3d(0.1,0.2,0.3));
    SE3 A = SE3(T);

    Eigen::Vector<double,6> twist1, twist2, twist3;

    twist1 << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0;
    twist2 << 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    twist3 << 0.0, 0.0, 1.0, 1.0, 0.0, 0.0;

    A.addExponential(twist1,0);
    A.addExponential(twist2,1);
    A.addExponential(twist3,2);

    Eigen::VectorXd theta(3);
    theta << 1.14, 2.15, 3.16;
    Eigen::Matrix4d v1 = A.eval(theta);
    Eigen::Matrix4d v2 = T.matrix() * se3Exp(theta(0) * twist1) * se3Exp(theta(1) * twist2) * se3Exp(theta(2) * twist3);
    REQUIRE( (v1 - v2).norm() < 1e-6);
}

TEST_CASE("SE3 * SE3", "SE3::operator*") {
    Eigen::Isometry3d Ta = Eigen::Isometry3d::Identity();
    Ta.rotate(Eigen::AngleAxisd(1.34,Eigen::Vector3d(0.1,0.2,0.3).normalized()));
    Ta.translate(Eigen::Vector3d(0.1,0.2,0.3));
    SE3 A = SE3(Ta);
    Eigen::Vector<double,6> atwist1;
    Eigen::Vector<double,6> atwist2;
    atwist1 << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    atwist2 << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    A.addExponential(atwist1,0);
    A.addExponential(atwist2,1);

    Eigen::Isometry3d Tb = Eigen::Isometry3d::Identity();
    Tb.rotate(Eigen::AngleAxisd(2.12,Eigen::Vector3d(-1.4,0.3,0.6).normalized()));
    Tb.translate(Eigen::Vector3d(0.8,-0.2,2.3));
    SE3 B = SE3(Tb);
    Eigen::Vector<double,6> btwist1;
    Eigen::Vector<double,6> btwist2;
    btwist1 << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    btwist2 << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    A.addExponential(btwist1,0);
    A.addExponential(btwist2,1);

    Eigen::VectorXd theta(2);
    theta << 0.1, 0.2;
    Eigen::Matrix4d v1 = (A * B).eval(theta);
    Eigen::Matrix4d v2 = A.eval(theta) * B.eval(theta);
    REQUIRE( (v1-v2).norm() < 1e-6);
}

TEST_CASE("SE3::evalVec","evalVec") {
    Eigen::Isometry3d Ta = Eigen::Isometry3d::Identity();
    Ta.rotate(Eigen::AngleAxisd(1.34,Eigen::Vector3d(0.1,0.2,0.3).normalized()));
    Ta.translate(Eigen::Vector3d(0.1,0.2,0.3));
    SE3 A = SE3(Ta);
    Eigen::Vector<double,6> atwist1;
    Eigen::Vector<double,6> atwist2;
    atwist1 << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    atwist2 << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    A.addExponential(atwist1,0);
    A.addExponential(atwist2,1);

    Eigen::Vector4d v;
    v << 0.1, 0.2, 0.3, 1.0;

    Eigen::VectorXd theta(2);
    theta << 0.34, 0.89;

    Eigen::Vector3d v1 = (A.eval(theta) * v).block<3,1>(0,0);
    Eigen::Vector3d v2 = A.evalVec(theta, v.block<3,1>(0,0));
    REQUIRE( (v1 - v2).norm() < 1e-6);
}

TEST_CASE("SE3::jacobian","jacobian") {
    SE3 A = SE3::translateParam(Eigen::Vector3d::UnitX(),0) * 
            SE3::translateParam(Eigen::Vector3d::UnitY(),1) *
            SE3::translateParam(Eigen::Vector3d::UnitZ(),2) *
            SE3::rotateZParam(5) *
            SE3::rotateYParam(4) *
            SE3::rotateXParam(3);

    Eigen::VectorXd theta(6);

    theta << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Eigen::MatrixXd J1 = A.jacobian(theta);
    Eigen::MatrixXd J2 = standardJacobian(theta);
    REQUIRE( (J1 - J2).norm() < 1e-6);

    for (int i = 0; i < 10; i++) {
        theta = Eigen::VectorXd::Random(6);
        Eigen::MatrixXd J3 = A.jacobian(theta);
        Eigen::MatrixXd J4 = standardJacobian(theta);
        REQUIRE( (J3 - J4).norm() < 1e-6);
    }
}
