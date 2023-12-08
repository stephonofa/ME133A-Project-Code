import rclpy
import numpy as np
from scipy.linalg import block_diag
 # Import the format for the condition number message
from std_msgs.msg import Float64
from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from pcode.AtlasGeneratorNode      import AtlasGeneratorNode, GeneratorNode
from pcode.TransformHelpers   import *
from pcode.TrajectoryUtils    import *

# Grab the general fkin from HW5 P5.
from pcode.KinematicChain     import KinematicChain

#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Setup up the condition number publisher
        #self.pub = node.create_publisher(Float64, '/condition', 10)
        
        # Initialize the transform broadcaster
        # self.broadcaster = TransformBroadcaster(self)

        # Add a publisher to send the joint commands.
        # self.pub = self.create_publisher(JointState, '/joint_states', 10)

        # # Wait for a connection to happen.  This isn't necessary, but
        # # means we don't start until the rest of the system is ready.
        # self.get_logger().info("Waiting for a /joint_states subscriber...")
        # while(not self.count_subscribers('/joint_states')):
        #     pass

        self.lf_chain = KinematicChain(node, 'pelvis', 'l_foot', self.lf_jointnames())
        self.rf_chain = KinematicChain(node, 'pelvis', 'r_foot', self.rf_jointnames())
        self.lh_chain = KinematicChain(node, 'pelvis', 'l_hand', self.lh_jointnames())
        self.rh_chain = KinematicChain(node, 'pelvis', 'r_hand', self.rh_jointnames())

        # Define the various points. Expand to be the total number of joints atlas has (30)
        self.q0 = np.zeros((len(self.jointnames()), 1))
        self.qlf = np.zeros((len(self.lf_jointnames()), 1))
        self.qrf = np.zeros((len(self.rf_jointnames()), 1))
        self.qlh = np.zeros((len(self.lh_jointnames()), 1))
        self.qrh = np.zeros((len(self.rh_jointnames()), 1))
        
        # Set the initial joint states for a push up in down position
        l_arm_shz = self.jointnames().index('l_arm_shz')
        r_arm_shz = self.jointnames().index('r_arm_shz')
        l_arm_shx = self.jointnames().index('l_arm_shx')
        r_arm_shx = self.jointnames().index('r_arm_shx')
        l_arm_ely = self.jointnames().index('l_arm_ely')
        r_arm_ely = self.jointnames().index('r_arm_ely')
        l_arm_elx = self.jointnames().index('l_arm_elx')
        r_arm_elx = self.jointnames().index('r_arm_elx')

        self.q0[l_arm_shz,0]     = -0.2
        self.q0[r_arm_shz,0]     =  0.2
        self.q0[l_arm_shx,0]     = -0.21
        self.q0[r_arm_shx,0]     =  0.21
        self.q0[l_arm_ely,0]     =  1.34
        self.q0[r_arm_ely,0]     =  1.34 
        self.q0[l_arm_elx,0]     =  1.34
        self.q0[r_arm_elx,0]     = -1.34
        # Initialize the current/starting joint position.
        self.q  = self.q0
        self.lam = 20
        self.qlf = self.q[:6]
        self.qrf = self.q[6:12]
        self.qlh = self.q[12:22]
        self.qrh = self.q[np.r_[12:15, 23:30]]

        # Set position/orientation of the pelvis (w.r.t. world).
        self.p_pelvis_0 = pxyz(0.0, 0.0, 0.47)
        self.R_pelvis_0 = Roty(19 * np.pi/48)
        self.p_pelvis = self.p_pelvis_0
        self.R_pelvis = self.R_pelvis_0

        # Calculate fixed position of feet and hands
        self.plh_fixed = self.p_pelvis + self.R_pelvis_0 @ self.lh_chain.fkin(self.qlh)[0]
        self.prh_fixed = self.p_pelvis + self.R_pelvis_0 @ self.rh_chain.fkin(self.qrh)[0]
        self.plf_fixed = self.p_pelvis + self.R_pelvis_0 @ self.lf_chain.fkin(self.qlf)[0]
        self.prf_fixed = self.p_pelvis + self.R_pelvis_0 @ self.rf_chain.fkin(self.qrf)[0]

        self.gamma = 1


    # joint names is a list of all joints
    # boradcast pelvis to fixed position whihch will serve as world
    # get push up and rotate pelvis
    def lf_jointnames(self):
        return ['l_leg_hpz', 'l_leg_hpx', 'l_leg_hpy', 'l_leg_kny', 'l_leg_aky', 'l_leg_akx']
    
    def rf_jointnames(self):
        return ['r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy', 'r_leg_kny', 'r_leg_aky', 'r_leg_akx']
    
    def lh_jointnames(self):
        return [
            'back_bkz', 'back_bky', 'back_bkx', 
            'l_arm_shz', 'l_arm_shx', 'l_arm_ely', 
            'l_arm_elx', 'l_arm_wry', 'l_arm_wrx', 'l_arm_wry2']
    
    def rh_jointnames(self):
        return [
            'back_bkz', 'back_bky', 'back_bkx', 
            'r_arm_shz', 'r_arm_shx', 'r_arm_ely', 
            'r_arm_elx', 'r_arm_wry', 'r_arm_wrx', 'r_arm_wry2']
    
    def hd_jointnames(self):
        return [
            'back_bkx', 'back_bky', 'back_bkz',
            'neck_ry',
        ]

    def jointnames(self):
        return self.lf_jointnames() + \
               self.rf_jointnames() + \
               self.hd_jointnames()[:3] + \
               self.lh_jointnames()[3:] + \
               ['neck_ry'] + \
               self.rh_jointnames()[3:]

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        vd_arms = pxyz(0.0, 0.0, -0.32 * cos(t))
        wd_arms = np.zeros((3,1))

        vd_feet = pxyz(0.0, 0.0, -0.32 * cos(t))
        wd_feet = np.zeros((3,1))

        pd_lh = np.linalg.inv(self.R_pelvis) @ (self.plh_fixed - self.p_pelvis)
        pd_rh = np.linalg.inv(self.R_pelvis) @ (self.prh_fixed - self.p_pelvis)
        pd_lf = np.linalg.inv(self.R_pelvis) @ (self.plf_fixed - self.p_pelvis)
        pd_rf = np.linalg.inv(self.R_pelvis) @ (self.prf_fixed - self.p_pelvis)

        qlast = self.q
        qlflast = qlast[:6]
        qrflast = qlast[6:12]
        qlhlast = qlast[12:22]
        qrhlast = qlast[np.r_[12:15, 23:30]]
        (Plf, Rlf, Jvlf, Jwlf) = self.lf_chain.fkin(qlflast)
        (Prf, Rrf, Jvrf, Jwrf) = self.rf_chain.fkin(qrflast)
        (Plh, Rlh, Jvlh, Jwlh) = self.lh_chain.fkin(qlhlast)
        (Prh, Rrh, Jvrh, Jwrh) = self.rh_chain.fkin(qrhlast)

        J_lf = np.vstack((Jvlf, Jwlf))
        J_rf = np.vstack((Jvrf, Jwrf))
        J_lh = np.vstack((Jvlh, Jwlh))
        J_rh = np.vstack((Jvrh, Jwrh))
        J_lh = np.hstack((J_lh, np.zeros((6, 8))))
        J_rh = np.hstack((J_rh[:, :3], np.zeros((6, 8)), J_rh[:, 3:]))
        J =  block_diag(J_lf, J_rf, np.vstack((J_lh, J_rh)))
        # print('\n\n\n\n:', J.shape, qlast.shape)
        # # Condition Number
        # Jbar = np.diag([1/0.4, 1/0.4, 1/0.4, 1, 1, 1]) @ J
        # condition = np.linalg.cond(Jbar)
        # # Publish the condition number.
        # msg = Float64()
        # msg.data = condition
        # self.pub.publish(msg)
        # construct qdot after getting desired positions and velocities
        Jweighted = np.linalg.inv(J @ J.T + self.gamma ** 2 * np.eye(24)) @ J
        qdot = np.linalg.pinv(Jweighted) @ (np.vstack((vd_feet, wd_feet, vd_feet, wd_feet, vd_arms, wd_arms, vd_arms, wd_arms)) +\
            self.lam * np.vstack((ep(pd_lf, Plf), eR(Reye(), Rlf), ep(pd_rf, Prf), eR(Reye(), Rrf), ep(pd_lh, Plh), eR(Reye(), Rlh), ep(pd_rh, Prh), eR(Reye(), Rrh))))

        q = qlast + qdot * dt
        self.q = q
        T_pelvis = T_from_Rp(self.R_pelvis, self.p_pelvis)

        # Compute position/orientation of the pelvis (w.r.t. world).
        new_p_pelvis = pxyz(0.0, 0.0, 0.32 * sin(t)) + self.p_pelvis_0
        new_R_pelvis = self.R_pelvis @ Roty(0.1 * sin(t))
        self.R_pelvis = new_R_pelvis
        self.p_pelvis = new_p_pelvis
        # Return the position and velocity as python lists.
        return (q.flatten().tolist(), qdot.flatten().tolist(), T_pelvis)


#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Initialize the generator node for 100Hz udpates, using the above
    # Trajectory class.
    generator = AtlasGeneratorNode('generator', 100, Trajectory)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

