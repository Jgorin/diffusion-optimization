
from diffusion_objective import DiffusionObjective
from diffusion_problem import DiffusionProblem
from jax import numpy as jnp
from dataset import Dataset
import torch as torch
import pandas as pd
import os

# get this file's directory
dir_path = os.path.dirname(os.path.realpath(__file__))

dataset = Dataset(pd.read_csv(f"{dir_path}/data/data4Optimizer.csv"))
objective = DiffusionObjective(
    dataset, 
    time_add = jnp.array([300*60,2003040*60]), 
    temp_add = jnp.array([40,21.111111111111]), 
    pickle_path = f"{dir_path}/data/lookup_table.pkl",
    omitValueIndices= [1,2,45]#[range(18,33)])
)

bounds = [
    (3996895114 - 3996895114*0.01 , 3996895114 +3996895114*0.01 ),
    (0.0001,150),
    (-10, 30), 
    (-10,30),
    (-10,30),
    (0.00001,1),
    (0.00001,1),
]

problem = DiffusionProblem(objective, bounds) 
problem.add_option('mu_strategy', 'adaptive')
problem.add_option('check_derivatives_for_naninf', 'yes')
problem.add_option('max_iter', 200)

x0 = jnp.array([0.5*(b[0]+b[1]) for b in bounds])
params, info = problem.solve(x0)
print(params)
