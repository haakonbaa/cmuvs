#include <cmuvs/util.h>

Eigen::Matrix4d wedge(const Eigen::Ref<const Eigen::Matrix<double,6,1>>& v) {
    Eigen::Matrix4d res = Eigen::Matrix4d::Zero();
    res.template block<3,3>(0,0) = skew(v.template block<3,1>(3,0).eval());
    res.template block<3,1>(0,3) = v.template block<3,1>(0,0);
    return res;
}

// [adjointMatrix(H) * v]_{wedge} = H [v]_{wedge} * H^{-1}
Eigen::Matrix<double, 6, 6> adjointMatrix(const Eigen::Isometry3d& T) {
    Eigen::Matrix3d R = T.rotation();
    Eigen::Vector3d p = T.translation();

    Eigen::Matrix<double,6,6> res = Eigen::Matrix<double,6,6>::Zero();
    res.block<3,3>(0,0) = R;
    res.block<3,3>(3,3) = R;
    res.block<3,3>(0,3) = skew(p) * R;

    return res;
}

// [adjointInvMatrix(H) * v]_{wedge} = H^{-1} [v]_{wedge} * H
Eigen::Matrix<double, 6, 6> adjointInvMatrix(const Eigen::Isometry3d& T) {
    Eigen::Matrix3d RT = T.rotation().transpose();
    Eigen::Vector3d p = T.translation();

    Eigen::Matrix<double,6,6> res = Eigen::Matrix<double,6,6>::Zero();
    res.block<3,3>(0,0) = RT;
    res.block<3,3>(3,3) = RT;
    res.block<3,3>(0,3) = - RT * skew(p);

    return res;
}
