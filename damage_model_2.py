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
import numpy as np
from sympy import Symbol, Eq

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.models.layers import Activation
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Circle, Triangle
from modulus.sym.domain.constraint import (PointwiseBoundaryConstraint, PointwiseInteriorConstraint)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.node import Node

# lib for time windows 
from modulus.sym.models.moving_time_window import MovingTimeWindowArch

# add inferencer
from modulus.sym.domain.inferencer import PointVTKInferencer
from modulus.sym.utils.io import (
    VTKUniformGrid,
)

# time variant solver
from modulus.sym.solver import SequentialSolver

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


"""
Neural Network design:

INPUT: ((x,y), t) {space-time coordinates}
OUTPUT: ((u,v), d) {extensions and damage}

Number of hidden layers and number of neurons to decided after tweaking with values

LE setup:

"""
# asumming 2D: planar stress state

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

    # Generate the sample geometry
    # coordinates for working field
    x, y, t = Symbol("x"), Symbol("y"), Symbol("t")

    panel_origin = (-.025, -.01) # (m, m) left bottom corner
    panel_dim = (0.05, 0.02) # (m, m)
    panel = Rectangle(panel_origin, (panel_origin[0] + panel_dim[0], panel_origin[1] + panel_dim[1]))

    tri = Triangle(panel_origin, .025, 0.001)  # (base = 25mm & height = 1 mm)

    thckness = 3e-3 #m

    # final geometry after remove those circle form the rectangular panel
    geo = panel - tri

    # bounds on the geometry
    bounds_x = (panel_origin[0], panel_origin[0] + panel_dim[0])
    bounds_y = (panel_origin[1], panel_origin[1] + panel_dim[1])

    # bound on time scale
    bounds_t = (0, 1.0)

    # time scales with we are working
    time_window_size = 0.1
    nr_time_windows = int(bounds_t[1]/time_window_size)
    time_range = {t : bounds_t}

    # setp model
    NN = instantiate_arch(
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
      
    time_window_net = MovingTimeWindowArch(NN, time_window_size)

    nodes = Equations.make_nodes() + [time_window_net.make_node(name="transient_time_net")]

    # DOMAIN
    ic_domain = Domain("Initial_conditions")
    win_domain = Domain("window")

    # Boundary conditions and Initial Conditions
    # left wall: no force / tractions, free surface
    panel_left = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"traction_x": 0.0, "traction_y": 0.0},
            batch_size = cfg.batch_size.panel_left,
            criteria = Eq(x, panel_origin[0]) & Eq(x/25 + y, -0.01), # walls where traction is 0
            parameterization={t: bounds_t, x:panel_origin[0]}, # t is a sympy symbol!!
            batch_per_epoch=500,
        )
    win_domain.add_constraint(panel_left, "panel_left")

    slant = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"traction_x": 0.0, "traction_y": 0.0},
            batch_size = cfg.batch_size.panel_left,
            criteria = Eq(x/25 + y, -0.01), # walls where traction is 0
            parameterization={t: bounds_t, x/25 + y: -.01}, # t is a sympy symbol!!
            batch_per_epoch=500,
        )
    win_domain.add_constraint(slant, "slant")
    
    panel_right = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"traction_x": 0.0, "traction_y": 0.0},
            batch_size = cfg.batch_size.panel_right,
            criteria = Eq(x, panel_origin[0] + panel_dim[0]), # defining the equations
            parameterization={t: bounds_t, x:(panel_origin[0] + panel_dim[0])}, # parametric terms
            batch_per_epoch=500,
        )
    win_domain.add_constraint(panel_right, "panel_right")

    # bottom wall
    panel_bottom = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar = {"v": 0.0, "d": 0.0},
            batch_size = cfg.batch_size.panel_bottom,
            criteria = Eq(y, panel_origin[1]) & (x >= panel_origin[0] + panel_dim[0] / 2.0),
            parameterization={t: bounds_t, y: panel_origin[1]}, # t is a sympy symbol!!
            batch_per_epoch=500,
        )
    win_domain.add_constraint(panel_bottom, "panel_bottom")

    # top wall
    # u_top = constant*t {the pseudo time constarint introduced here}
    panel_top = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"v": u_max*t},
            batch_size = cfg.batch_size.panel_top,
            criteria = Eq(y, panel_origin[1] + panel_dim[1]),
            parameterization={t: bounds_t, y:(panel_origin[1] + panel_dim[1])},
            batch_per_epoch=500,
            
        )
    win_domain.add_constraint(panel_top, "panel_top")

    # initial condition @t = 0 => d = 0
    IC = PointwiseInteriorConstraint(
        nodes = nodes,
        geometry = geo,
        outvar={"d": 0.0},
        batch_size=cfg.batch_size.lr_interior,
        # criteria = Eq(t, bounds_t[0]),
        # bounds={x: bounds_x, y: bounds_y},
        lambda_weighting={"d": 10.0},
        parameterization={t : 0.0},
        batch_per_epoch=500,
    )
    ic_domain.add_constraint(IC, "IC")

    # constarints to be hold: interior
    Interior = PointwiseInteriorConstraint(
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
        bounds={x: bounds_x, y: bounds_y, t: bounds_t},
        lambda_weighting={
            "equilibrium_x": 10.0,
            "equilibrium_y": 10.0, # Symbol("sdf") = default
            "stress_disp_xx": 1.0,
            "stress_disp_yy": 1.0,
            "stress_disp_xy": 1.0,
            "DM": 5.0,
        },
        parameterization=time_range,
        batch_per_epoch=500,
    )
    win_domain.add_constraint(Interior, "governing_equations_windows")
    ic_domain.add_constraint(Interior, "governing_equations_interior")

    nx = 500
    ny = 200

    # adding the time_inferencer as we need to store the time varied results
    # add inference data for time slices
    for i, specific_time in enumerate(np.linspace(0, time_window_size, nr_time_windows)):
        vtk_obj = VTKUniformGrid(
            # give the geometric bounds below
            bounds=[bounds_x, bounds_y],
            # set the required number of points
            npoints=[nx, ny],
            # export map for the same
            export_map={"U": ["u", "v"], "D": ["d"], "F":["f"]},
        )
        grid_inference = PointVTKInferencer(
            vtk_obj=vtk_obj,
            nodes=nodes,
            input_vtk_map={"x": "x", "y": "y"},
            output_names=["u", "v", "d", "f"],
            # requires_grad=False,
            # create the time varible at instant that would be constant
            invar={"t": np.full([nx*ny, 1], specific_time)},
            batch_size=1000,
        )
        ic_domain.add_inferencer(grid_inference, name = "time = {} , slice = {}".format(str(specific_time), str(i).zfill(4)))
        win_domain.add_inferencer(grid_inference, name = "time = {} , slice = {}".format(str(specific_time), str(i).zfill(4)))

    # make solver
    slv = SequentialSolver(cfg, [(1, ic_domain), (nr_time_windows, win_domain)], custom_update_operation = time_window_net.move_window)
    
    # start solver
    slv.solve()

if __name__ == "__main__":
    run()