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
# from modulus.sym.eq.pdes.linear_elasticity import LinearElasticityPlaneStress

from phi_beta_model import Model2D
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
                Key("t")],
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
    x, y, t = Symbol("x"), Symbol("y"), Symbol("t")
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
    panel_left = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"traction_x": 0.0, "traction_y": 0.0},
            batch_size = cfg.batch_size.panel_left,
            criteria = Eq(x, panel_origin[0]), # equations to be need to defined here
            parameterization={t: bounds_t, x:panel_origin[0]}, # t is a sympy symbol!!
            batch_per_epoch=500,
        )
    domain.add_constraint(panel_left, "panel_left")
    
    panel_right = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"traction_x": 0.0, "traction_y": 0.0},
            batch_size = cfg.batch_size.panel_right,
            criteria = Eq(x, panel_origin[0] + panel_dim[0]), # defining the equations
            parameterization={t: bounds_t, x:(panel_origin[0] + panel_dim[0])}, # parametric terms
            batch_per_epoch=500,
        )
    domain.add_constraint(panel_right, "panel_right")

    # bottom wall
    panel_bottom = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar = {"v": 0.0, "d": 0.0},
            batch_size = cfg.batch_size.panel_bottom,
            criteria = Eq(y, panel_origin[1]),
            parameterization={t: bounds_t, y: panel_origin[1]}, # t is a sympy symbol!!
            batch_per_epoch=500,
        )
    domain.add_constraint(panel_bottom, "panel_bottom")

    # top wall
    # u_top = constant*t {the pseudo time constarint introduced here}
    panel_top = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"v": u_max*t},
            batch_size = cfg.batch_size.panel_top,
            criteria = Eq(y, panel_origin[1] + panel_dim[1]),
            parameterization={t: bounds_t, y:(panel_origin[1] + panel_dim[1])}, # t is a sympy symbol!!
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
        },
        batch_size=cfg.batch_size.lr_interior,
        bounds={x: bounds_x, y: bounds_y, t: bounds_t},
        lambda_weighting={
            "equilibrium_x": Symbol("sdf"),
            "equilibrium_y": Symbol("sdf"),
            "stress_disp_xx": Symbol("sdf"),
            "stress_disp_yy": Symbol("sdf"),
            "stress_disp_xy": Symbol("sdf"),
        },
        parameterization={t: bounds_t, x: bounds_x, y: bounds_y},
        batch_per_epoch=500,
    )
    domain.add_constraint(LE, "LE_interior")

    DM1 = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"f1":0},
        criteria= Equations.equations["p"]<r1, # and Equations.equations["p"]> -r1,
        batch_size=cfg.batch_size.lr_interior,
        bounds={x: bounds_x, y: bounds_y, t: bounds_t},
        lambda_weighting = {"f1": 1.0},
        parameterization={t: bounds_t, x: bounds_x, y: bounds_y},
        batch_per_epoch=500,
    ) 
    
    domain.add_constraint(DM1, "damage_model_1")

    # DM2 = PointwiseInteriorConstraint(
    #     nodes=nodes,
    #     geometry=geo,
    #     outvar={"f2":0},
    #     criteria= Equations.equations["p"]>=r1 and Equations.equations["p"]=<-r1,
    #     batch_size=cfg.batch_size.lr_interior,
    #     bounds={x: bounds_x, y: bounds_y, t: bounds_t},
    #     lambda_weighting = {"f2": 1.0},
    #     parameterization={t: bounds_t, x: bounds_x, y: bounds_y},
    #     batch_per_epoch=500,
    # ) 
    
    # domain.add_constraint(DM2, "damage_model_2")

    # initial condition @t = 0 => d = 0
    IC = PointwiseInteriorConstraint(
        nodes = nodes,
        geometry = geo,
        outvar={"d": 0},
        batch_size=cfg.batch_size.lr_interior,
        criteria = Eq(t, bounds_t[0]),
        bounds={x: bounds_x, y: bounds_y},
        lambda_weighting={"d": 1.0},
        parameterization={x:bounds_x, y:bounds_y},
        batch_per_epoch=500,
    )
    domain.add_constraints(IC, "IC")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()
