import gurobipy as grb
import numpy as np
import copy

RLP = grb.Model('Relaxed MIP')

var_index = [
    f'x_{i}'
    for i in range(1, 3)
]
x_dict = RLP.addVars(var_index, vtype=grb.GRB.CONTINUOUS, name= var_index)

RLP.addConstr(2 * x_dict['x_1'] + x_dict['x_2'] <=10, name= 'c1')
RLP.addConstr(3 * x_dict['x_1'] + 6 * x_dict['x_2'] <=40, name='c2')
RLP.setObjective(100 * x_dict['x_1'] + 150 * x_dict['x_2'], grb.GRB.MAXIMIZE)
RLP.optimize()