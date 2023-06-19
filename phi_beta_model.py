"""
Equations used from Prof. Junker model, Institute of Continum mechanics, Leibniz University Hannover

Author: Nihal Pushkar
email: nihalpushkar11@gmail.com; me1200947@iitd.ac.in
Affiliated: Department of Mechanical Engineering, Indian Institute of Technology Delhi
Host: Summer Internship at Institute of Continum mechanics, Leibniz University Hannover

"""

import numpy as np
from sympy import Symbol, Function, Max, Number, log, Abs, simplify
from modulus.sym.eq.pde import PDE
from modulus.sym.node import Node

class Model2D(PDE):
    # 2D stress case if assumed here
    def __init__(self, E=None, nu=None, lambda_=None, mu=None, rho=1, beta=0, flag=0, r1=1000, r2=0):

        # spacial coordinates
        x, y = Symbol("x"), Symbol("y")
        normal_x, normal_y = Symbol("normal_x"), Symbol("normal_y")

        # time coordinates
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "t": t}

        # displacement componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)

        # stress components predicted
        sigma_xx = Function("sigma_xx")(*input_variables)
        sigma_yy = Function("sigma_yy")(*input_variables)
        sigma_xy = Function("sigma_xy")(*input_variables)

        # set equations
        self.equations = {}

        # Stress equations
        w_z = -lambda_ / (lambda_ + 2 * mu) * (u.diff(x) + v.diff(y))
        self.equations["stress_disp_xx"] = (lambda_ * (u.diff(x) + v.diff(y) + w_z) + 2 * mu * u.diff(x) - sigma_xx)
        
        self.equations["stress_disp_yy"] = (lambda_ * (u.diff(x) + v.diff(y) + w_z) + 2 * mu * v.diff(y) - sigma_yy)
        self.equations["stress_disp_xy"] = mu * (u.diff(y) + v.diff(x)) - sigma_xy

        # Equations of equilibrium: COM
        self.equations["equilibrium_x"] = rho * ((u.diff(t)).diff(t)) - (sigma_xx.diff(x) + sigma_xy.diff(y))
        self.equations["equilibrium_y"] = rho * ((v.diff(t)).diff(t)) - (sigma_xy.diff(x) + sigma_yy.diff(y))

        # Traction equations
        self.equations["traction_x"] = normal_x * sigma_xx + normal_y * sigma_xy
        self.equations["traction_y"] = normal_x * sigma_xy + normal_y * sigma_yy
        
        # Linear Elasticity: Energy equation
        # strains
        e_xx = u.diff(x)
        e_yy = v.diff(y)
        e_xy = 0.5*(v.diff(x) + u.diff(v))

        # phi_{0}
        self.equations["phi_0"] = 0.5*(sigma_xx*e_xx + sigma_yy*e_yy + 2*sigma_xy*e_xy)
         
        # damage variable
        d = Function("d")(*input_variables)
        
        self.equations["d"] = d

        if flag == 0:
            # exponential
            f = 1 - np.exp(-1*d)
        else:
            # linear
            f = 1 - d

        # /phi
        self.equations["phi"] = f*self.equations["phi_0"] + 0.5*beta*((f.diff(x))**2 + (f.diff(y))**2)

        # /delta^{diss}
        self.equations["diss"] = r1*(f.diff(t)) + 0.5*r2*(f.diff(t))**2

        # p
        self.equations["p"] = -1*self.equations["phi_0"] + beta*((f.diff(x)).diff(x) + (f.diff(y)).diff(y))
        # print(type(self.equations["p"]))
        
        # the equation below is: max(|p| - r_{1}, 0) + r_{2}*f_dot = 0
        # evolution equation from the vitual work principal, which is put to zero condionally
        
        # self.equations["f"] = self.equations["p"] - r1 + r2*(f.diff(t))
        # self.equations["f"] = Max((Max(self.equations["p"], -1*self.equations["p"]) - r1), 0) - r1 + r2*(f.diff(t))

        # DM1
        self.equations["f1"] = f.diff(t)

        # DM2
        self.equations["f2"] = self.equations["p"] - r1 + r2*(f.diff(t))

