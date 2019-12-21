import numpy as np
from numpy.matlib import repmat
from numpy import zeros, eye, ones, sqrt, asscalar

# from cvxopt import solvers, matrix

# class SafeSet(MobileAgent):
class SafeSet:

    """
    This is the Safe Set method. Please refer to the paper for details.
    """

    def __init__(self, d_min=20, k_v=2, yita=10):

        self.lambd = 0.5 # uncertainty margin
        self.half_plane_ABC = []
        
        self.safe_set = [0,0,0]
        self.k_v = k_v
        self.d_min = d_min
        self.yita = yita


    # def calc_control_input(self, dT, goal, fx, fu, Xr, Xh, dot_Xr, dot_Xh, Mr, Mh, p_Mr_p_Xr, p_Mh_p_Xh, u0, min_u, max_u):
    def calc_control_input(self, fx, fu, dot_Xh, Mr, Mh, p_Mr_p_Xr, p_Mh_p_Xh, u0, dT: float = .1):
        # delta T = 0.1, according to multi_agent_env default simulation time
        dim = np.shape(Mr)[0] // 2
        # print("dim:", dim)
        p_idx = np.arange(dim)
        # print("p_idx:", p_idx)
        v_idx = p_idx + dim
        # print("v_idx:", v_idx)

        d = np.linalg.norm(Mr[p_idx] - Mh[p_idx])
        print("d:", d)
        
        # sgn = -1 if np.asscalar((Mr[[0,1],0] - Mh[[0,1],0]).T * (Mr[[2,3],0] - Mh[[2,3],0])) < 0 else 1
        # dot_d = sgn * sqrt((Mr[2,0] - Mh[2,0])**2 + (Mr[3,0] - Mh[3,0])**2)

        # dot_Mr = p_Mr_p_Xr * dot_Xr
        # dot_Mh = p_Mh_p_Xh * dot_Xh

        dM = Mr - Mh
        # dot_dM = dot_Mr - dot_Mh
        # dp = dM[p_idx,0]
        # dv = dM[v_idx,0]
        dp = dM[p_idx]
        dv = dM[v_idx]
        # print("dM:", dM)
        # print("dp:", dp)
        # print("dv:", dv)

        # dot_dp = dot_dM[p_idx,0]
        # dot_dv = dot_dM[v_idx,0]

        #dot_d is the component of velocity lies in the dp direction
        dot_d = dp.T * dv / d

        p_dot_d_p_dp = dv / d - asscalar(dp.T * dv) * dp / (d**3)
        p_dot_d_p_dv = dp / d
        # print("p_dot_d_p_dp:", p_dot_d_p_dp)
        # print("p_dot_d_p_dv:", p_dot_d_p_dv)
        
        p_dp_p_Mr = np.hstack([eye(dim), zeros((dim,dim))])
        p_dp_p_Mh = -p_dp_p_Mr
        # print("p_dp_p_Mr:", p_dp_p_Mr)
        # print("p_dp_p_Mh:", p_dp_p_Mh)

        p_dv_p_Mr = np.hstack([zeros((dim,dim)), eye(dim)])
        p_dv_p_Mh = -p_dv_p_Mr
        # print("p_dv_p_Mr:", p_dv_p_Mr)
        # print("p_dv_p_Mh:", p_dv_p_Mh)

        p_dot_d_p_Mr = p_dp_p_Mr.T * p_dot_d_p_dp + p_dv_p_Mr.T * p_dot_d_p_dv
        p_dot_d_p_Mh = p_dp_p_Mh.T * p_dot_d_p_dp + p_dv_p_Mh.T * p_dot_d_p_dv

        # print("p_dot_d_p_Mr:", p_dot_d_p_Mr)
        # print("p_dot_d_p_Mh:", p_dot_d_p_Mh)

        p_dot_d_p_Xr = p_Mr_p_Xr.T * p_dot_d_p_Mr
        p_dot_d_p_Xh = p_Mh_p_Xh.T * p_dot_d_p_Mh

        # print("p_dot_d_p_Xr:", p_dot_d_p_Xr)
        # print("p_dot_d_p_Xh:", p_dot_d_p_Xh)


        d = 1e-3 if d == 0 else d
        dot_d = 1e-3 if dot_d == 0 else dot_d

        p_d_p_Mr = np.vstack([ dp / d, zeros((dim,1))])
        p_d_p_Mh = np.vstack([-dp / d, zeros((dim,1))])

        p_d_p_Xr = p_Mr_p_Xr.T * p_d_p_Mr
        p_d_p_Xh = p_Mh_p_Xh.T * p_d_p_Mh

        # print("p_d_p_Mr:", p_d_p_Mr)
        # print("p_d_p_Mh:", p_d_p_Mh)
        # print("p_d_p_Xr:", p_d_p_Xr)
        # print("p_d_p_Xh:", p_d_p_Xh)
        

        phi = self.d_min**2 + self.yita * dT + self.lambd * dT - d**2 - self.k_v * dot_d;
        # phi = self.d_min**2 - d**2 - self.k_v * dot_d;

        p_phi_p_Xr = - 2 * d * p_d_p_Xr - self.k_v * p_dot_d_p_Xr;
        p_phi_p_Xh = - 2 * d * p_d_p_Xh - self.k_v * p_dot_d_p_Xh;

        # dot_phi = p_phi_p_Xr.T * dot_Xr + p_phi_p_Xh.T * dot_Xh;
        print("phi:", phi)

        # print("u:", u0)
        # print("p_phi_p_Xr:", p_phi_p_Xr)
        # print("p_phi_p_Xh:", p_phi_p_Xh)


        L = p_phi_p_Xr.T * fu;
        S = - self.yita - self.lambd - p_phi_p_Xh.T * dot_Xh - p_phi_p_Xr.T * fx;
        # print("p_phi_p_Xr.T * fx =", asscalar(p_phi_p_Xr.T * fx))
        # print("p_phi_p_Xh.T * dot_Xh =", asscalar(p_phi_p_Xh.T * dot_Xh))


        # print("p_phi_p_Xh:", p_phi_p_Xh)
        # print("L:", L)
        # print("S:", S)
        # print("u:", u0)
        
        u = u0;
        print("L * u = {}, and S = {}".format(asscalar(L * u0), asscalar(S)))
        if phi <= -self.yita * 0.4 or asscalar(L * u0) < asscalar(S):
            # print("====== NO SSA =======")
            u = u0;
        else:
            print("====== SSA TRIGGERED ======")
            try:
                # Q = matrix(eye(np.shape(u0)[0]))
                # p = matrix(- 2 * u0)
                # nu = np.shape(u0)[0]
                # G = matrix(np.vstack([eye(nu), -eye(nu)]))
                # r = matrix(np.vstack([max_u, -min_u]))
                # A = matrix([[matrix(L),G]])
                # b = matrix([[matrix(S),r]])
                # solvers.options['show_progress'] = False
                # sol=solvers.qp(Q, p, A, b)
                # u = np.vstack(sol['x'])
                u = u0 - (asscalar(L * u0 - S) * L.T / asscalar(L * L.T));
            except ValueError:
                # print('no solution')
                u = u0 - (asscalar(L * u0 - S) * L.T / asscalar(L * L.T));
            pass
            # 

        A = asscalar(L[0,0]);
        B = asscalar(L[0,1]);
        C = asscalar(S[0,0]);

        # self.half_plane_ABC = np.matrix([A, B, -C - A * Xr[0] - B * Xr[1]])
        # self.ABC = np.matrix([A, B, C])

        self.fuck = False

        return u


    def eval_phi(self, fx, fu, dot_Xh, Mr, Mh, p_Mr_p_Xr, p_Mh_p_Xh, u0, dT: float = .1):
        dim = np.shape(Mr)[0] // 2
        # print("dim:", dim)
        p_idx = np.arange(dim)
        # print("p_idx:", p_idx)
        v_idx = p_idx + dim
        # print("v_idx:", v_idx)

        d = np.linalg.norm(Mr[p_idx] - Mh[p_idx])
        # print("d:", d)
        
        # sgn = -1 if np.asscalar((Mr[[0,1],0] - Mh[[0,1],0]).T * (Mr[[2,3],0] - Mh[[2,3],0])) < 0 else 1
        # dot_d = sgn * sqrt((Mr[2,0] - Mh[2,0])**2 + (Mr[3,0] - Mh[3,0])**2)

        # dot_Mr = p_Mr_p_Xr * dot_Xr
        # dot_Mh = p_Mh_p_Xh * dot_Xh

        dM = Mr - Mh
        # dot_dM = dot_Mr - dot_Mh
        # dp = dM[p_idx,0]
        # dv = dM[v_idx,0]
        dp = dM[p_idx]
        dv = dM[v_idx]

        #dot_d is the component of velocity lies in the dp direction
        dot_d = dp.T * dv / d
        phi = self.d_min**2 + self.yita * dT + self.lambd * dT - d**2 - self.k_v * dot_d
        phi = np.asscalar(phi)
        return phi