import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer, functions, load_model_from_xml
from scipy.optimize import minimize


""" 
TODO: BROKEN CODE, NEED TO FIX
"""
""" 
    There are three sites defined as of now. 1. 'debrisSite': at the COM of debris, 2. 'baseSite': at the COM of base.
    3. 'end_effector': at the middle of the two fingers. 
    BELOW WORKS ONLY AFTER CALLING STEP() AT LEAST ONCE
    sim.data.site_xpos : gives the cartesian position of all the sites
    sim.data.site_xmat : gives the rotation matrix of all the sites
                            OR
    sim.data.get_site_xpos(name)

"""


# void mj_jac(const mjModel* m, const mjData* d,
#                       mjtNum* jacp, mjtNum* jacr, const mjtNum point[3], int body)
# while t<tlim:
#     sim.data.qpos[:] = pos[t,:]
#     sim.data.qvel[:] = vel[t,:]
#     sim.data.qacc[:] = acc[t,:]
#     functions.mj_inverse(sim.model, sim.data)
#     torque_est[t,:] = sim.data.qfrc_inverse
#     t += 1
#     env.sim.step()


class InvKinSpaceRobot:
    def inv_kin(self, ee_target, jt_pos):
        # model, data = self.model.copy(), self.data.copy()
        # data.qpos = jt_pos
        # functions.mj_kinematics(model, data)
        self.target = ee_target
        res = minimize(self.f,
                       x0=jt_pos,
                       method="SLSQP",
                       options={'maxiter': 500, 'disp': True}, jac=self.g)
        # print(res)
        return res.x

    def get_jacobian(self, site=0):
        # site : 0 for left hand gripper
        ## return translation jacobian of the first site

        jac = np.zeros((3, 7))
        idx = site * 3
        temp = np.zeros((3, 26 + 6 - 7))

        # functions.mj_jacSite(env.model.ptr, env.data.ptr, temp.ctypes.data_as(POINTER(c_double)), None, site)
        functions.mj_jacSite(self.model.uintptr, self.data.uintptr, temp, None, site)
        functions.g
        # set_trace()
        jac = temp[:, 10:17]
        # print('jac')
        # print(jac)
        # print("----")
        return jac

    def apply_action(self, action={"left": None, "right": None}):

        ctrl = self.data.ctrl.ravel().copy()
        if len(action["left"]) > 0:
            ctrl[7:14] = np.array(action["left"])
        if len(action["right"]) > 0:
            ctrl[:7] = np.array(action["right"])
        for i, c in enumerate(ctrl):
            self.data.ctrl[i] = c

    def set_params(self, x):
        self.apply_action(action={"right": [], "left": x})
        for _ in range(2000):
            self.step()

    def f(self, x):
        self.set_params(x)

        lhs = self.data.site_xpos[0]
        rhs = self.target
        cost = 0.5 * np.linalg.norm(lhs - rhs) ** 2
        # print("cost:%.4f"%cost)
        # set_trace()
        return cost

    def g(self, x):
        self.set_params(x)

        lhs = env.data.site_xpos[0]
        rhs = self.target

        J = self.get_jacobian(site=0)
        g = (lhs - rhs)[np.newaxis, :].dot(J).flatten()
        return g

    def do_ik(self, ee_target, jt_pos):
        self.target = ee_target
        res = minimize(self.f,
                       x0=jt_pos,
                       method="SLSQP",
                       options={'maxiter': 500, 'disp': True}, jac=self.g)
        # print(res)
        return res.x