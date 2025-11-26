'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''


from pid import PIDAgent
from keyframes import hello, leftBackToStand
from typing import Tuple, List, Any, Dict



class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        #target_joints['RHipYawPitch'] = target_joints['LHipYawPitch'] # copy missing joint in keyframes
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)




    def angle_interpolation(self, keyframes: Tuple[List[Any], List[Any], List[Any]], perception):
        target_joints: Dict[str, float] = {}
        names, times, keys = keyframes
        t = perception.time

        def cubic_bezier_1d(p0, p1, p2, p3, u: float) -> float:
            """Standard cubic Bézier in 1D for parameter u ∈ [0, 1]."""
            x = 1.0 - u
            return (
                x**3 * p0
                + 3.0 * x**2 * u * p1
                + 3.0 * x * u**2 * p2
                + u**3 * p3
            )

        for j, name in enumerate(names):
            joint_times = times[j]      # list of times for this joint
            joint_keys = keys[j]        # list of [angle, handle1, handle2] for this joint

            if not joint_times:
                continue

            # before first keyframe → hold first angle
            if t <= joint_times[0]:
                target_joints[name] = joint_keys[0][0]
                continue

            # after last keyframe → hold last angle
            if t >= joint_times[-1]:
                target_joints[name] = joint_keys[-1][0]
                continue

            # find segment i with t_i <= t <= t_{i+1}
            for i in range(len(joint_times) - 1):
                t0, t1 = joint_times[i], joint_times[i + 1]
                if t0 <= t <= t1:
                    k0 = joint_keys[i]
                    k1 = joint_keys[i + 1]

                    angle0 = k0[0]
                    angle1 = k1[0]

                    # handles: second handle of k0 (curve after k0),
                    #          first handle of k1 (curve before k1)
                    h2 = k0[2]        # [type, dTime, dAngle]
                    h1_next = k1[1]   # [type, dTime, dAngle]

                    dAngle_right = h2[2]
                    dAngle_left = h1_next[2]

                    # 1D control "points" in angle space
                    p0 = angle0
                    p1 = angle0 + dAngle_right
                    p2 = angle1 + dAngle_left
                    p3 = angle1

                    # normalize current time to [0,1] within this segment
                    u = (t - t0) / (t1 - t0)

                    angle = cubic_bezier_1d(p0, p1, p2, p3, u)
                    target_joints[name] = angle
                    break  # done with this joint

        return target_joints

if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
