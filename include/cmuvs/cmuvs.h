#pragma once
#include <Eigen/Dense>

// A Wrench is a collection of a force and a torque that is applied to a
// position.
struct Wrench {
    Eigen::Vector3d position;
    Eigen::Vector6d wrench;
    unsigned int inputVariableIndex;
}

struct Link {
    double mass;
    double volume;
    Eigen::Matrix3d inertia;          // Inertia matrix defined in center of mass
    Eigen::Matrix6d addedMass;        // Added mass in link center
    Eigen::Matrix6d linearDamping;    // Linear damping in link center
    Eigen::Matrix6d quadraticDamping; // Quadratic damping in link center
    Eigen::Vector3d centerOfMass;     // Center of mass relative to link center
    Eigen::Vector3d centerOfBuoyancy; // Center of buoyancy relative to link center
    std::vector<Wrench> wrenches;     // List of wrenches acting on Link
};

class Eely {
    std::vector<Link> links;
}

enum class BaseLinkTransform {
    quaternion,
    EulerXYZ
}

class Robot {
public:
    Robot(Link baseLink, BaseLinkTransform baseLinkTransform, std::vector<std::pair<Link, SE3>>)
    {

    }
private:
}
