"""
Developing the python script for the model proposed by Prof. Junker, IKM, LUH:
https://arxiv.org/abs/2102.08819

Period: June 2023

Author: Nihal Pushkar & Tobias Bode

Author: Nihal Pushkar
email: nihalpushkar11@gmail.com; me1200947@iitd.ac.in
Affiliated: Department of Mechanical Engineering, Indian Institute of Technology Delhi
Host: Summer Internship 2023 at Institute of Continum mechanics, Leibniz University Hannover

The aim is model the governing equations as depicted in the damage model specific for the geometry given.
"""

# import the dependicies
from ast import And
from sympy import Symbol, Eq

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.models.layers import Activation
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Circle
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.utils.io.vtk import VTKUniformGrid
from modulus.sym.domain.inferencer import PointVTKInferencer

import numpy as np
from sympy import Symbol, Function, Max, Number, log, Abs, simplify
from modulus.sym.eq.pde import PDE
from modulus.sym.node import Node

# from modulus.sym.eq.pdes.linear_elasticity import LinearElasticityPlaneStress
"""
Custom model specific for out problem

PDE solving classes are to scripted here
"""

# material specifics
E = 70e9 # Pa; young's modulus
nu = 0.3 # poisons ratio

# model specifics
beta = 10 # N; damaage capturing term
r1 = 1e4 # Pa; Linear term for /delta^{diss}
r2 = 1e4 # Pa-s; Qudratic term for /delta^{diss}

"""
Working variables: Rate dependent gradient {in exp decay of damage}
/phi_{0} = elatic energy stored

/phi = total energy for 

/delta^{diss} = damage dissapiation term {Pa/s}

f = damage function

/beta = regularization term
"""

class Model2D(PDE):
    # 2D stress case if assumed here
    def __init__(self, E=None, nu=None, lambda_=None, mu=None, rho=1, beta=0, flag=0, r1=1000, r2=0):

        # spacial coordinates
        x, y = Symbol("x"), Symbol("y")
        normal_x, normal_y = Symbol("normal_x"), Symbol("normal_y")

        # time coordinates
        z = Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}

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
        self.equations["equilibrium_x"] = rho * ((u.diff(z)).diff(z)) - (sigma_xx.diff(x) + sigma_xy.diff(y))
        self.equations["equilibrium_y"] = rho * ((v.diff(z)).diff(z)) - (sigma_xy.diff(x) + sigma_yy.diff(y))

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
        self.equations["d"] = 0.5*(d + (d**2)**(0.5)) # ensuring always positive
        
        # self.equations["test"] = self.equations["d"] + (self.equations["d"]**2)**0.5

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
        self.equations["diss"] = r1*(f.diff(z)) + 0.5*r2*(f.diff(z))**2

        # p
        self.equations["p"] = -1*self.equations["phi_0"] + beta*((f.diff(x)).diff(x) + (f.diff(y)).diff(y))

        # p_hat
        self.equations["p_hat"] = self.equations["p"] - r1

        # p_plus
        self.equations["p_plus"] = 0.5*(self.equations["p_hat"] + (self.equations["p_hat"]**2)**0.5)

        # DM
        self.equations["DM"] = r2*f.diff(z) + self.equations["p_plus"]


# create run; setup the config filesconda 
@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))  # Pa
    mu_real = E / (2 * (1 + nu))  # Pa
    lambda_ = lambda_ / mu_real  # Dimensionless
    mu = 1.0  # Dimensionless

    u_max = 0.01 # m, say for the sake of example here

    # Using LE for now
    Equations = Model2D(E = E, nu = nu, lambda_=lambda_, mu = mu, rho = 1, beta = beta, flag = 1, r1 = r1, r2 = r2)

    """
    Neural Network design:

    INPUT: ((x,y), t) {space-time coordinates}
    OUTPUT: ((u,v), d) {extensions and damage}

    Number of hidden layers and number of neurons to decided after tweaking with values

    LE setup:

    """
    # asumming 2D: planar stress state
    net = instantiate_arch(
            # INPUTs
            input_keys=[
                Key("x"), 
                Key("y"), 
                Key("z")],
            # OUTPUTs
            output_keys=[
                Key("u"),
                Key("v"),
                Key("sigma_xx"),
                Key("sigma_yy"),
                Key("sigma_xy"),
                Key("d"),
            ],
            cfg=cfg.arch.fully_connected, # change size from the config.yaml
            activation_fn=Activation.TANH,
            
        )
    nodes = Equations.make_nodes() + [net.make_node(name="virtual_work_model")]

    # Generate the sample geometry
    # coordinates for working field
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    time_range = {z : (0, 1)}
    panel_origin = (-.02, -.055) # (m, m) left bottom corner
    panel_dim = (0.04, 0.11) # (m, m)
    circle_radius = 0.005 # (m)
    circle1_origin = (-.02, -.01) # (m, m)
    circle2_origin = (.02, .01) # (m, m)
    panel = Rectangle(panel_origin, (panel_origin[0] + panel_dim[0], panel_origin[1] + panel_dim[1]))
    c1 = Circle(circle1_origin, circle_radius)
    c2 = Circle(circle2_origin, circle_radius)
    thckness = 3e-3 #m

    # final geometry after remove those circle form the rectangular panel
    geo = panel - c1 - c2

    # geometry 2 in the paper:
    rec_origin = (0, 0) # (m, m)
    u0 = 1e-3 # m

    # bounds on the geometry
    bounds_x = (panel_origin[0], panel_origin[0] + panel_dim[0])
    bounds_y = (panel_origin[1], panel_origin[1] + panel_dim[1])

    # bound on time scale
    bounds_t = (0, 1)

    # DOMAIN
    domain = Domain()

# Boundary conditions and Initial Conditions
    # left wall: no force / tractions, free surface
    panel_left_1 = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"traction_x": 0.0, "traction_y": 0.0},
            batch_size = cfg.batch_size.panel_left,
            criteria = Eq(x, panel_origin[0]), # equations to be need to defined here of constarints
            parameterization={z: bounds_t, x:panel_origin[0]}, # t is a sympy symbol!!
            batch_per_epoch=500,
        )
    domain.add_constraint(panel_left_1, "panel_left_1")

    panel_left_2 = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"traction_x": 0.0, "traction_y": 0.0},
            batch_size = cfg.batch_size.panel_left,
            criteria = Eq((x - circle1_origin[0])**2 + (y - circle1_origin[1])**2, circle_radius**2), # equations to be need to defined here of constarints
            parameterization={z: bounds_t, (x - circle1_origin[0])**2 + (y - circle1_origin[1])**2 : circle_radius**2}, # t is a sympy symbol!!
            batch_per_epoch=500,
        )
    domain.add_constraint(panel_left_2, "panel_left_2")

    # right wall
    panel_right_1 = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"traction_x": 0.0, "traction_y": 0.0},
            batch_size = cfg.batch_size.panel_right,
            criteria = Eq(x, panel_origin[0] + panel_dim[0]), # defining the equations (constraints)
            parameterization={z: bounds_t, x:(panel_origin[0] + panel_dim[0])}, # parametric terms
            batch_per_epoch=500,
        )
    domain.add_constraint(panel_right_1, "panel_right_1")

    panel_right_2 = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"traction_x": 0.0, "traction_y": 0.0},
            batch_size = cfg.batch_size.panel_right,
            criteria = Eq((x - circle2_origin[0])**2 + (y - circle2_origin[1])**2, circle_radius**2), # defining the equations (constraints)
            parameterization={z: bounds_t, (x - circle2_origin[0])**2 + (y - circle2_origin[1])**2 : circle_radius**2}, # parametric terms
            batch_per_epoch=500,
        )
    domain.add_constraint(panel_right_2, "panel_right_2")

    # bottom wall
    panel_bottom = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar = {"v": 0.0, "d": 0.0},
            batch_size = cfg.batch_size.panel_bottom,
            criteria = Eq(y, panel_origin[1]),
            parameterization={z: bounds_t, y: panel_origin[1]}, # t is a sympy symbol!!
            batch_per_epoch=500,
        )
    domain.add_constraint(panel_bottom, "panel_bottom")

    # top wall
    # u_top = constant*t {the pseudo time constarint introduced here}
    panel_top = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"v": u_max*z},
            batch_size = cfg.batch_size.panel_top,
            criteria = Eq(y, panel_origin[1] + panel_dim[1]),
            parameterization={z: bounds_t, y:(panel_origin[1] + panel_dim[1])},
            batch_per_epoch=500,
            
        )
    domain.add_constraint(panel_top, "panel_top")

    # inferenceer if required

    # constarints to be hold: interior
    LE = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "equilibrium_x": 0.0,
            "equilibrium_y": 0.0,
            "stress_disp_xx": 0.0,
            "stress_disp_yy": 0.0,
            "stress_disp_xy": 0.0, 
            "DM":0.0,
        },
        batch_size=cfg.batch_size.lr_interior,
        bounds={x: bounds_x, y: bounds_y, z: bounds_t},
        lambda_weighting={
            "equilibrium_x": 1.0,
            "equilibrium_y": 1.0, # Symbol("sdf")
            "stress_disp_xx": 1.0,
            "stress_disp_yy": 1.0,
            "stress_disp_xy": 1.0,
            "DM": 1.0,
        },
        parameterization={z: bounds_t, x: bounds_x, y: bounds_y},
        batch_per_epoch=500,
    )
    domain.add_constraint(LE, "interior")

    # inferencer
    vtk_obj = VTKUniformGrid(
        bounds=[bounds_x, bounds_y, bounds_t],
        npoints=[40, 110, 10],
        export_map={"U": ["u", "v", None], "F": ["f"], "D" : {"d"}},
    )
    grid_inference = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=nodes,
        input_vtk_map={"x": "x", "y": "y", "z":"z"},
        output_names=["u", "v", "f", "d"],
        requires_grad=False,
        batch_size=1024,
    )
    domain.add_inferencer(grid_inference, "inferencer")

    # initial condition @t = 0 => d = 0
    IC = PointwiseInteriorConstraint(
        nodes = nodes,
        geometry = geo,
        outvar={"d": 0.0},
        batch_size=cfg.batch_size.lr_interior,
        # criteria = Eq(t, bounds_t[0]),
        # bounds={x: bounds_x, y: bounds_y},
        lambda_weighting={"d": 1.0},
        parameterization={z : 0.0},
        batch_per_epoch=500,
    )
    domain.add_constraint(IC, "IC")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()
