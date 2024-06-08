import gurobipy as grb
import numpy as np
import copy
import os

"""
This py file is to help create a base Model that can be used to create its relaxation model in an automatic way

Define->Optimize->Save

All you need to do is to define: 
- model_name: give a model name.
- model_var: the adding model var function that need to return the var space to model_CnO to add contraint and set objective.
- model_CnO: the function to add constraint and set obejective.
- save_path: the path you want to save the model expression and its solution.

An example that defines model_var and model_CnO is set as default.
"""
def create_model(model_name='Model', model_var=add_var, model_CnO=add_CnO_simple_IP_model_01, save_path=r'.\test_model'):
    created_model = grb.Model(model_name)
    
    # add var
    
    
    # add constraint and set objective
#     add_CnO_simple_IP_model_01(created_model, add_var(created_model))
    model_CnO(model=created_model, x_dict=model_var(created_model))
    
    created_model.optimize()
    save_ModelnSolution(created_model, save_path)
    return created_model
   
def add_var(model):
    var_index = [
    f'x_{i}'
    for i in range(1, 3)
    ]
    x_dict = model.addVars(var_index, vtype=grb.GRB.INTEGER, name= var_index)
    return x_dict

def add_CnO_simple_IP_model_01(model, x_dict):    
    model.addConstr(2 * x_dict['x_1'] + x_dict['x_2'] <=10, name= 'c1')
    model.addConstr(3 * x_dict['x_1'] + 6 * x_dict['x_2'] <=40, name='c2')
    model.setObjective(100 * x_dict['x_1'] + 150 * x_dict['x_2'], grb.GRB.MAXIMIZE)

def save_ModelnSolution(model, path):
    model.write(os.path.join(path, model.ModelName+'.lp'))
    model.write(os.path.join(path, model.ModelName+'.sol'))
