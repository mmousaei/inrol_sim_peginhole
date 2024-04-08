#include "ros/ros.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/WrenchStamped.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float64.h"
#include "your_package_name/TargetPoseAndRoll.h"  // Update this with the actual package and message name
#include "inrol_sim_peginhole/inrol_sim_peginhole.hpp"
#include "inrol_sim_peginhole_main/msg_for_joint.h"

// Pub
ros::Publisher obj_pose_pub;
ros::Publisher contact_wrench_pub;
ros::Publisher joint_angle_pub;

geometry_msgs::PoseStamped msg_obj_pose;
geometry_msgs::WrenchStamped msg_contact_wrench;
inrol_sim_peginhole_main::msg_for_joint msg_joint_angle;

// Sub
ros::Subscriber target_sub;

// Simulation variables
double xk_sim[3];
double qk_sim[4]; // Quaternion: x, y, z, w
double ak_sim[8];
double Fc_sim[6];

// Received target
bool target_received = false;
geometry_msgs::Point target_position;
double target_roll;

void targetCallback(const your_package_name::TargetPoseAndRoll& msg)
{
    target_position = msg.position;
    target_roll = msg.roll;
    target_received = true;
}

void publish()
{
    msg_obj_pose.pose.position.x = xk_sim[0];
    msg_obj_pose.pose.position.y = xk_sim[1];
    msg_obj_pose.pose.position.z = xk_sim[2];
    msg_obj_pose.pose.orientation.x = qk_sim[0];
    msg_obj_pose.pose.orientation.y = qk_sim[1];
    msg_obj_pose.pose.orientation.z = qk_sim[2];
    msg_obj_pose.pose.orientation.w = qk_sim[3];
    msg_obj_pose.header.stamp = ros::Time::now();

    msg_contact_wrench.wrench.force.x = Fc_sim[0];
    msg_contact_wrench.wrench.force.y = Fc_sim[1];
    msg_contact_wrench.wrench.force.z = Fc_sim[2];
    msg_contact_wrench.wrench.torque.x = Fc_sim[3];
    msg_contact_wrench.wrench.torque.y = Fc_sim[4];
    msg_contact_wrench.wrench.torque.z = Fc_sim[5];
    msg_contact_wrench.header.stamp = ros::Time::now();

    for (size_t j = 0; j < 7; j++)
    {
        msg_joint_angle.joint_angle[j] = ak_sim[j];
    }

    obj_pose_pub.publish(msg_obj_pose);						  
    contact_wrench_pub.publish(msg_contact_wrench);	  
    joint_angle_pub.publish(msg_joint_angle);	 
}

void Callback_start_bool(const std_msgs::Bool msg_start_bool)
{
	start_bool =  msg_start_bool.data;
}


void convertRollToQuaternion(double roll, double quaternion[])
{
    // Assuming roll is about the Z-axis and pitch, yaw are zero
    double cy = cos(roll * 0.5);
    double sy = sin(roll * 0.5);

    quaternion[0] = 0; // x
    quaternion[1] = 0; // y
    quaternion[2] = sy; // z
    quaternion[3] = cy; // w
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "simulation_main");
    ros::NodeHandle n("~");
    ros::Rate loop_rate(100);

    obj_pose_pub = n.advertise<geometry_msgs::PoseStamped>("/obj_pose",100);
	contact_wrench_pub = n.advertise<geometry_msgs::WrenchStamped>("/contact_wrench",100);
	joint_angle_pub = n.advertise<inrol_sim_peginhole_main::msg_for_joint>("/franka_joint_angle",100);


    // Subscribe to target pose and roll topic
    target_sub = n.subscribe("/target_pose_and_roll", 10, targetCallback);
    start_bool_sub = n.subscribe("/start_bool",20, Callback_start_bool);

    while (ros::ok())
    {
        ros::spinOnce();

        if (target_received)
        {
            double Rd[3][3]; // This will need to be computed based on the target roll
            double pd[3] = {target_position.x, target_position.y, target_position.z};
            double quat[4];
            convertRollToQuaternion(target_roll, quat);
            // Note: You'll need to convert quat to Rd if your sim_function requires a rotation matrix

            inrol_sim::sim_function(pd, Rd, xk_sim, qk_sim, ak_sim, Fc_sim);
            publish();
            target_received = false; // Reset flag
        }

        loop_rate.sleep();
    }

    return 0;
}
