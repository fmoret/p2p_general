# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:11:54 2018

@author: fmoret
"""

# Import Gurobi Library
import gurobipy as gb
import numpy as np

# Class which can have attributes set.
class expando(object):
    pass


# Subproblem
class Prosumer:
    def __init__(self, inc, data=None, rho=1):
        self.data = expando()
        self.data.a = data['a']
        self.data.b = data['b']
        self.data.Pmin = data['Pmin']
        self.data.Pmax = data['Pmax']
        self.data.num_assets = data['num_ass']
        self.data.partners = inc.nonzero()
        self.data.partners_free = inc[:,:2].nonzero()
        self.data.partners_pos = inc[:,2:].nonzero()
        self.data.pref = inc[inc.nonzero()]
        self.data.pref_free = inc[:,:2][inc[:,:2].nonzero()]
        self.data.pref_pos = inc[:,2:][inc[:,2:].nonzero()]
        self.data.num_partners = len(self.data.pref)
        self.data.num_partners_free = len(self.data.pref_free)
        self.data.num_partners_pos = len(self.data.pref_pos)
        for i in range(self.data.num_partners_pos):
            self.data.partners_pos[1][i] += 2
        self.data.rho = rho
        
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._build_model()

    def optimize(self, trade):
        self._iter_update(trade)
        self._update_objective()
        self.model.optimize()
        for i in range(self.data.num_partners):
            self.t_old[i] = self.t[i].x
        trade[self.data.partners] = self.t_old
        return trade

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self.model.setParam( 'OutputFlag', False )
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.model.update()

    def _build_variables(self):
        m = self.model
        self.variables.p = np.array([m.addVar(lb = self.data.Pmin[i], ub = self.data.Pmax[i], name = 'p') for i in range(self.data.num_assets)])
        self.variables.t_free = np.array([m.addVar(lb = -gb.GRB.INFINITY, name = 't_free') for i in range(self.data.num_partners_free)])
        self.variables.abs_t = np.array([m.addVar(obj=self.data.pref_free[i], name = 'abs_t') for i in range(self.data.num_partners_free)])
        self.variables.t_pos = np.array([m.addVar(obj=self.data.pref_pos[i], name = 't_pos') for i in range(self.data.num_partners_pos)])
        self.t = np.append(self.variables.t_free, self.variables.t_pos)
        self.t_old = np.zeros(self.data.num_partners)
        self.y = np.zeros(self.data.num_partners)
        m.update()
        
    def _build_constraints(self):
        self.constraints.pow_bal = self.model.addConstr(sum(self.variables.p) == sum(self.variables.t_free) + sum(self.variables.t_pos))
        for i in range(self.data.num_partners_free):
            self.model.addConstr(self.variables.t_free[i] <= self.variables.abs_t[i])
            self.model.addConstr(self.variables.t_free[i] >= -self.variables.abs_t[i])
        
    def _build_objective(self):
        self.obj_assets = sum(self.data.b*self.variables.p + self.data.a*self.variables.p*self.variables.p)
        
    ###
    #   Model Updating
    ###    
    def _update_objective(self):
        augm_lag = (-sum(self.y*( self.t - self.t_average ) ) + 
                    self.data.rho/2*sum( ( self.t - self.t_average )
                                        *( self.t - self.t_average ) )
                   )
        self.model.setObjective(self.obj_assets + augm_lag)
        self.model.update()
        
    ###
    #   Iteration Update
    ###    
    def _iter_update(self, trade):
        temp = np.append(trade[self.data.partners_free], trade[self.data.partners_pos])
        self.t_average = (self.t_old - temp)/2
        self.y -= self.data.rho*(self.t_old - self.t_average)
#        for i in range(self.data.num_partners):
#            self.y[i] +=  self.rho*(self.t_old[i] + self.t_others[i])/2