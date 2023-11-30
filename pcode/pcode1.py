
import rclpy
import numpy as np
from scipy.linalg import block_diag
 # Import the format for the condition number message
from std_msgs.msg import Float64
from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from hw5code.GeneratorNode      import GeneratorNode
from hw5code.TransformHelpers   import *
from hw5code.TrajectoryUtils    import *

# Grab the general fkin from HW5 P5.
from hw5code.KinematicChain     import KinematicChain
from tf2_ros                    import TransformBroadcaster
from geometry_msgs.msg          import TransformStamped
from sensor_msgs.msg            import JointState

#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Setup up the condition number publisher
        #self.pub = node.create_publisher(Float64, '/condition', 10)
        
        # # Initialize the transform broadcaster
        # self.broadcaster = TransformBroadcaster(self)

        # # Add a publisher to send the joint commands.
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
        #self.rf_chain = KinematicChain(node, 'world', 'tip', self.jointnames())

        # Define the various points. Expand to be the total number of joints atlas has (30)
        self.q0 = np.zeros((len(self.jointnames()), 1))
        self.qlf = np.zeros((len(self.lf_jointnames()), 1))
        self.qrf = np.zeros((len(self.rf_jointnames()), 1))
        self.qlh = np.zeros((len(self.lh_jointnames()), 1))
        self.qrh = np.zeros((len(self.rh_jointnames()), 1))

        # Initialize the current/starting joint position.
        self.q  = self.q0
        self.lam = 20
        #TODO Attach pelvis to world frame
        p_pelvis = pxyz(0.0, 0.3, 0.0)
        #TODO set the initial joint states for a push up
        # lh_relbow = self.jointnames().index('l_arm_shx')
        # rh_relbow = self.jointnames().index('r_arm_shx')
        # lt_relbow = self.jointnames().index('l_arm_ely')
        # rt_relbow = self.jointnames().index('r_arm_ely')

        # self.q[lh_relbow,0]     = - pi/2
        # self.q[rh_relbow,0]     =  pi/2
        # self.q[lt_relbow,0]     =  pi/2 * 0
        # self.q[rt_relbow,0]     =  -pi/2 * 0

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
               self.hd_jointnames() + \
               self.lh_jointnames() + \
               self.rh_jointnames()

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
    
        # # Compute position/orientation of the pelvis (w.r.t. world).
        # ppelvis = pxyz(0.0, 0.5, 1.5 + 0.5 * sin(t/2))
        # Rpelvis = Rotz(sin(t))
        # Tpelvis = T_from_Rp(Rpelvis, ppelvis)
        
        # # Build up and send the Pelvis w.r.t. World Transform!
        # trans = TransformStamped()
        # trans.header.frame_id = 'world'
        # trans.child_frame_id  = 'pelvis'
        # trans.transform       = Transform_from_T(Tpelvis)
        # self.broadcaster.sendTransform(trans)

        qlast = self.q
        (Plf, Rlf, Jvlf, Jwlf) = self.lf_chain.fkin(self.qlf)
        (Prf, Rrf, Jvrf, Jwrf) = self.rf_chain.fkin(self.qrf)
        (Plh, Rlh, Jvlh, Jwlh) = self.lh_chain.fkin(self.qlh)
        (Prh, Rrh, Jvrh, Jwrh) = self.rh_chain.fkin(self.qrh)

        J_lf = np.vstack((Jvlf, Jwlf))
        J_rf = np.vstack((Jvrf, Jwrf))
        J_lh = np.vstack((Jvlh, Jwlh))
        J_rh = np.vstack((Jvrh, Jwrh))
        J_lh =np.hstack((J_lh, np.zeros((6, 8))))
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
        #TODO construct qdot after getting our two positions and creating a spline
        qdot = np.zeros((len(self.jointnames()), 1))
        # np.linalg.pinv(J) @ (np.vstack((vd, wd)) + self.lam * np.vstack((ep(pd, P), eR(Rd, R))))

        q = qlast + qdot * dt
        self.q = q
        # Return the position and velocity as python lists.
        return (q.flatten().tolist(), qdot.flatten().tolist())


#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Initialize the generator node for 100Hz udpates, using the above
    # Trajectory class.
    generator = GeneratorNode('generator', 100, Trajectory)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

