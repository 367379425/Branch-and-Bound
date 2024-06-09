import gurobipy as grb
import numpy as np
import copy
import os

eps = 1e-3
class Global_BB:
    def __init__(self, model):
        self.gub = np.inf
        self.glb = 0
        self.node_pool = []
        
        self.incumbent_solution = None
        self.best_objective = 0
        
        self.MIP_var_vtype_dict = {var.varName: var.vtype for var in model.getVars()}
#         {var_name: 'I' for var_name in [f'x_{i}' for i in range(1, 3)]}
#         {var.varName: var.vtype for var in model.getVars()}
        
        self.model_relaxation = model.copy() 
        for var in self.model_relaxation.getVars():
            var.vtype = grb.GRB.CONTINUOUS
        self.model_relaxation.update()        
        self.root_node = Node(self.model_relaxation, self.MIP_var_vtype_dict, '0')
        
        print('root node begin optimizing')
        self.root_node.optimize_lpr(self.glb)
        print('root optimization finished')
        self.incumbent_solution = self.root_node.x_int_solution
        self.best_objective = self.root_node.node_LB
        self.gub = self.root_node.node_UB
        self.glb = self.root_node.node_LB
        print(f"x_sol:\t{self.root_node.x_solution}\nincumbent_solution:\t{self.incumbent_solution}\nbest_objective:\t{self.best_objective}\nGlobal_UB:\t{self.gub}\nGlobal_LB:\t{self.glb}")
        
        if self.root_node.is_pruned == 0:
            self.node_pool.append(self.root_node)
            self.gub_log = [self.gub]
            self.glb_log = [self.glb]
            print('root node is not punred')
    
    def BandB(self):
#         if len(self.node_pool) == 0:
#             return 
#         node_id = 1
        while len(self.node_pool)>0 and self.gub - self.glb > eps:
            current_node = self.node_pool.pop(self.estimate_node())
            print(f"node: {current_node.node_id} is selected to branch")
            temp = current_node.branch(self)
            printChildNode_info(current_node)
            self.node_pool += temp
            if len(temp) > 0:
                self.gub = max([node.node_UB for node in self.node_pool])
            self.glb_log.append(self.glb)
            self.gub_log.append(self.gub)
            # 更新 全局上下界
            
    
    def estimate_node(self):
#         if len(self.node_pool) == 0:
#             return 1 # no unbranched or pruned node 
        
        # choose the node with greatest node_UB
        best_i = max([
            (i, self.node_pool[i].node_UB)
            for i in range(len(self.node_pool))
        ], key=lambda x: x[-1])[0]
        
        return best_i
    

class Node:
    def __init__(self, model, MIP_var_vtype_dict, node_id):
        self.node_id = node_id
        self.node_LB = 0
        self.node_UB = np.inf
        
        self.x_solution = None
        self.x_int_solution = None
        
        self.MIP_var_vtype_dict = MIP_var_vtype_dict # no relaxation model's var's vtype C for CONTINUOUS; B for BINARY I for INTEGER
        self.branch_var_dict = None
        
        self.model = model
        self.model_status = False
        
        self.is_integer = None
        self.is_pruned = 0 # pruned 1: by integer solution; 2: by bounded 3: by infeasible or unbound
        
        self.branched = False
        
#         self.optimize_lpr()

    
    def optimize_lpr(self, glb):
        self.model.setParam('OutputFlag', 0)
        self.model.update()
        
        # optimize the node model and intialize the node param
        self.model.optimize()
        """
        OPTIMAL = 2
        INFEASIBLE = 3
        UNBOUNDED = 5
        """
        self.model_status = self.model.Status
        
        if self.model_status == 2: # have optimal solution
            self.node_UB = self.model.ObjVal
            
            self.x_solution = {var.varName: var.x for var in self.model.getVars()}
            self.x_int_solution = {var.varName: int(var.x) if self.MIP_var_vtype_dict[var.varName] in ['B', 'I'] else var.x
                                   for var in self.model.getVars()}
#             print(f"x_solution: {self.x_solution}")
#             print(f"x_int_solution: {self.x_int_solution}")
            
            self.node_LB = sum([
                self.x_int_solution[var.varName] * var.Obj
                for var in self.model.getVars()
            ])
            
            
            self.branch_var_dict = {
                var.varName: 1 if (self.MIP_var_vtype_dict[var.varName] in ['B', 'I']
                and abs(self.x_solution[var.varName] - self.x_int_solution[var.varName])>eps) # 1: if a binary or integer var is no integer value; Need to branch
                else 0 # 0: ① if a binary or integer var is integer value; ② continuous var. No Need to branch
                for var in self.model.getVars()
                }
            self.is_integer = 0 if 1 in self.branch_var_dict.values() else 1 # if 1 in branch_var_dict means it still have binary or integer var that is no integer value
            self.PruneByOptimility()
            self.PruneByBounded(glb)
#             self.PruneByOptimility()
#             self.PruneByBounded()
            
        else: # INFEASIBLE or UNBOUNDED      
            self.PrunedByUnfeasibleOrUnbound()
    
    def branch(self, global_BB):
        
        branch_var_list = list(filter(lambda x: self.branch_var_dict[x]==1, self.branch_var_dict))
        # most infeasible branching, 选择小数部分最接近0.5的进行分支
        self.branch_var_name = min(
                [(var_name, abs(self.x_solution[var_name] - self.x_int_solution[var_name])) 
                 for var_name in branch_var_list], key=lambda x : abs(x[-1]-0.5)
        )[0]
        
        # ①BINARY 0, 1. xr=0.xxx  left var BOUND = 0, right_var BOUND =1
        # ②CONTINOUS. xr=A.BBB  left var BOUND= A right var bound = A+1
        
        print(f'branch_var_name:{self.branch_var_name}')
        left_var_bound = int(self.x_solution[self.branch_var_name])
        right_var_bound = left_var_bound+1
        
        # create two child nodes
        self.left_node, self.right_node = self.deepcopy_node(node_id=self.node_id+'->l'), self.deepcopy_node(node_id=self.node_id+'->r')
#         self.left_node.node_id += '->l'
#         self.right_node.node_id += '->r'
        
        # add constraint for left
        branch_var = self.left_node.model.getVarByName(self.branch_var_name)
        self.left_node.model.addConstr(branch_var <= left_var_bound, name='branch_left')
        self.left_node.optimize_lpr(global_BB.glb)
        check_update(global_BB, self.left_node)
    
        
        # add constraint for right
        branch_var = self.right_node.model.getVarByName(self.branch_var_name)
        self.right_node.model.addConstr(branch_var >= right_var_bound, name='branch_right')
        self.right_node.optimize_lpr(global_BB.glb)
        check_update(global_BB, self.right_node)

        
        temp = []
        if self.left_node.is_pruned == 0:
            temp.append(self.left_node)
        if self.right_node.is_pruned == 0:
            temp.append(self.right_node)
        return temp
          
    
    def PruneByOptimility(self):
        if self.is_integer == 1:
#             print('Pruned by optimility')
            self.is_pruned = 1
            
    def PruneByBounded(self, global_lb):
        if self.node_UB < global_lb:
#             print('Pruned by bound')
            self.is_pruned = 2
    
    def PrunedByUnfeasibleOrUnbound(self):
#         print('Pruned by unfeasible or unbound')
        self.is_pruned = 3
    
    def deepcopy_node(self, node_id):
        new_node = Node(self.model.copy(), self.MIP_var_vtype_dict, node_id)
        return new_node

def check_update(global_, local):

    if local.is_pruned != 0: # if node is pruned, pass
        return
    if local.node_LB > global_.glb:
        global_.glb = local.node_LB
        global_.incumbent_solution = local.x_int_solution
        global_.best_objective = local.node_LB
#         return 1

def printChildNode_info(parent):
    is_pruned_dict = {1: 'Pruned by optimility', 2: 'Pruned by bound', 3: 'Pruned by unfeasible or unbound'}
    print(f"node: {parent.node_id} \'child node info:")
    if parent.left_node.is_pruned != 0:
        print(f"left_node: {parent.left_node.node_id} is {is_pruned_dict[parent.left_node.is_pruned]}")
    print(f"left_node {parent.left_node.node_id} info:")
    printNodeInfo(parent.left_node)
    
    if parent.right_node.is_pruned != 0:
        print(f"right_node: {parent.right_node.node_id} is {is_pruned_dict[parent.right_node.is_pruned]}")
    print(f"right_node {parent.right_node.node_id} info:")
    printNodeInfo(parent.right_node)

def printNodeInfo(node):
    print(f"x_sol:\t{node.x_solution}")
    print(f"x_int_sol:\t{node.x_int_solution}")
    print(f"UB:\t{node.node_UB}")
    print(f"LB:\t{node.node_LB}")
    