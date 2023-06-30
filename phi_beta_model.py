"""
Equations used from Prof. Junker model, Institute of Continum mechanics, Leibniz University Hannover

Author: Nihal Pushkar
email: nihalpushkar11@gmail.com; me1200947@iitd.ac.in
Affiliated: Department of Mechanical Engineering, Indian Institute of Technology Delhi
Host: Summer Internship at Institute of Continum mechanics, Leibniz University Hannover
"""

import numpy as np
from sympy import Symbol, Function, Number, log, Abs, simplify
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

        # material properties; in case not mentioned thus adding (E, /nu) to outputs of the neural network
        if lambda_ is None:
            if isinstance(nu, str):
                nu = Function(nu)(*input_variables)
            elif isinstance(nu, (float, int)):
                nu = Number(nu)
            if isinstance(E, str):
                E = Function(E)(*input_variables)
            elif isinstance(E, (float, int)):
                E = Number(E)
            lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
        else:
            if isinstance(lambda_, str):
                lambda_ = Function(lambda_)(*input_variables)
            elif isinstance(lambda_, (float, int)):
                lambda_ = Number(lambda_)
            if isinstance(mu, str):
                mu = Function(mu)(*input_variables)
            elif isinstance(mu, (float, int)):
                mu = Number(mu)
        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # make input variables
        input_variables = {"x": x, "y": y, "t": t}

        # Symbols
        u, v = Symbol("u"), Symbol("v")
        sigma_xx, sigma_xy, sigma_yy = Symbol("sigma_xx"), Symbol("sigma_xy"), Symbol("sigma_yy")

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
        e_xy = 0.5*(v.diff(x) + u.diff(y))

        # phi_{0}
        self.equations["phi_0"] = 0.5*(sigma_xx*e_xx + sigma_yy*e_yy + 2*sigma_xy*e_xy)
        
        # damage variable
        d = Function("d")(*input_variables)
        self.equations["d"] = 0.5*(d + (d**2)**(0.5)) # ensuring always positive

        if flag == 0:
            # exponential
            # f = 1 - (np.exp(2*self.equations["d"]) - 1)/(np.exp(2*self.equations["d"]) + 1) # f = 1 - tanh(d)
            f = 1 - np.exp(-1*self.equations["d"])
        else:
            # linear
            f = 1 - d

        self.equations["f"] = f # damage function

        # /phi
        self.equations["phi"] = f*self.equations["phi_0"] + 0.5*beta*((f.diff(x))**2 + (f.diff(y))**2)

        # /delta^{diss}
        self.equations["diss"] = r1*(f.diff(t)) + 0.5*r2*(f.diff(t))**2

        # p
        self.equations["p"] = -1*self.equations["phi_0"] + beta*((f.diff(x)).diff(x) + (f.diff(y)).diff(y))

        # the equation below is: max(|p| - r_{1}, 0) + r_{2}*f_dot = 0
        # evolution equation from the vitual work principal, which is put to zero condionally
        # type casting the equations from <sympy.core.add.Add> => <sympy.core.pow.Pow>: ensures the integretity of functional forms of the equations

        # p_hat
        self.equations["p_hat"] = self.equations["p"] - r1

        # p_plus
        self.equations["p_plus"] = 0.5*(self.equations["p_hat"] + (self.equations["p_hat"]**2)**0.5)

        # DM
        self.equations["DM"] = r2*f.diff(t) + self.equations["p_plus"]

class Model3D(PDE):
    # more geenric case
    def __init__(self, E=None, nu=None, lambda_=None, mu=None, rho=1, beta=0, flag=0, r1=1000, r2=0):

        # spacial coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # plane defination
        normal_x, normal_y, normal_z = (
            Symbol("normal_x"),
            Symbol("normal_y"),
            Symbol("normal_z"),
        )

        # time coordinate
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}

        # displacement componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        w = Function("w")(*input_variables)

        # stress components
        sigma_xx = Function("sigma_xx")(*input_variables)
        sigma_yy = Function("sigma_yy")(*input_variables)
        sigma_xy = Function("sigma_xy")(*input_variables)
        sigma_zz = Function("sigma_zz")(*input_variables)
        sigma_xz = Function("sigma_xz")(*input_variables)
        sigma_yz = Function("sigma_yz")(*input_variables)

        # material properties
        if lambda_ is None:
            if isinstance(nu, str):
                nu = Function(nu)(*input_variables)
            elif isinstance(nu, (float, int)):
                nu = Number(nu)
            if isinstance(E, str):
                E = Function(E)(*input_variables)
            elif isinstance(E, (float, int)):
                E = Number(E)
            lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
        else:
            if isinstance(lambda_, str):
                lambda_ = Function(lambda_)(*input_variables)
            elif isinstance(lambda_, (float, int)):
                lambda_ = Number(lambda_)
            if isinstance(mu, str):
                mu = Function(mu)(*input_variables)
            elif isinstance(mu, (float, int)):
                mu = Number(mu)
        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # making the set of equations
        self.equations = {}

        # governing equations
        # Stress equations
        self.equations["stress_disp_xx"] = (lambda_ * (u.diff(x) + v.diff(y) + w.diff(z)) + 2 * mu * u.diff(x) - sigma_xx)
        self.equations["stress_disp_yy"] = (lambda_ * (u.diff(x) + v.diff(y) + w.diff(z)) + 2 * mu * v.diff(y) - sigma_yy)
        self.equations["stress_disp_zz"] = (lambda_ * (u.diff(x) + v.diff(y) + w.diff(z)) + 2 * mu * w.diff(z) - sigma_zz)
        self.equations["stress_disp_xy"] = mu * (u.diff(y) + v.diff(x)) - sigma_xy
        self.equations["stress_disp_xz"] = mu * (u.diff(z) + w.diff(x)) - sigma_xz
        self.equations["stress_disp_yz"] = mu * (v.diff(z) + w.diff(y)) - sigma_yz

        # Equations of equilibrium
        self.equations["equilibrium_x"] = rho * ((u.diff(t)).diff(t)) - (sigma_xx.diff(x) + sigma_xy.diff(y) + sigma_xz.diff(z))
        self.equations["equilibrium_y"] = rho * ((v.diff(t)).diff(t)) - (sigma_xy.diff(x) + sigma_yy.diff(y) + sigma_yz.diff(z))
        self.equations["equilibrium_z"] = rho * ((w.diff(t)).diff(t)) - (sigma_xz.diff(x) + sigma_yz.diff(y) + sigma_zz.diff(z))

        # Navier equations
        self.equations["navier_x"] = (rho * ((u.diff(t)).diff(t))
            - (lambda_ + mu) * (u.diff(x) + v.diff(y) + w.diff(z)).diff(x)
            - mu * ((u.diff(x)).diff(x) + (u.diff(y)).diff(y) + (u.diff(z)).diff(z))
        )
        self.equations["navier_y"] = (rho * ((v.diff(t)).diff(t))
            - (lambda_ + mu) * (u.diff(x) + v.diff(y) + w.diff(z)).diff(y)
            - mu * ((v.diff(x)).diff(x) + (v.diff(y)).diff(y) + (v.diff(z)).diff(z))
        )
        self.equations["navier_z"] = (rho * ((w.diff(t)).diff(t))
            - (lambda_ + mu) * (u.diff(x) + v.diff(y) + w.diff(z)).diff(z)
            - mu * ((w.diff(x)).diff(x) + (w.diff(y)).diff(y) + (w.diff(z)).diff(z))
        )

        # damage variable
        d = Function("d")(*input_variables)
        self.equations["d"] = 0.5*(d + (d**2)**(0.5)) # ensuring always positive

        # strains
        e_xx = u.diff(x)
        e_yy = v.diff(y)
        e_zz = w.diff(z)
        e_xy = 0.5*(v.diff(x) + u.diff(y))
        e_xz = 0.5*(w.diff(x) + u.diff(z))
        e_yz = 0.5*(v.diff(z) + w.diff(y))

        # /phi_0
        self.equations["phi_0"] = 0.5*(sigma_xx*e_xx + sigma_yy*e_yy + 2*sigma_xy*e_xy + sigma_zz*e_zz + 2*sigma_xz*e_xz + 2*sigma_yz*e_yz)

        if flag == 0:
            # exponential
            # f = 1 - (np.exp(2*self.equations["d"]) - 1)/(np.exp(2*self.equations["d"]) + 1) # f = 1 - tanh(d)
            f = 1 - np.exp(-1*self.equations["d"])
        else:
            # linear
            f = 1 - d

        self.equations["f"] = f # damage function

        # /phi
        self.equations["phi"] = f*self.equations["phi_0"] + 0.5*beta*((f.diff(x))**2 + (f.diff(y))**2 + (f.diff(z))**2)

        # /delta^{diss}
        self.equations["diss"] = r1*(f.diff(t)) + 0.5*r2*(f.diff(t))**2

        # p
        self.equations["p"] = -1*self.equations["phi_0"] + beta*((f.diff(x)).diff(x) + (f.diff(y)).diff(y) + (f.diff(z)).diff(z))

        # p_hat
        self.equations["p_hat"] = self.equations["p"] - r1

        # p_plus
        self.equations["p_plus"] = 0.5*(self.equations["p_hat"] + (self.equations["p_hat"]**2)**0.5)

        # DM
        self.equations["DM"] = r2*f.diff(t) + self.equations["p_plus"]
