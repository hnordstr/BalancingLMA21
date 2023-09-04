"""This is the optimization problem where imbalance netting occurs"""

import gurobipy as gp
from gurobipy import GRB
from gurobipy import *


class GurobiModel:

    def __init__(self, name):
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            self.gm = gp.Model(name, env=env)
        #self.gm = gp.Model(name)
        self.name = name

    def setup_problem(self, m):
        """ PM = parent model """
        self.pm = m
        # self.pm.create_highres()
        print('SETTING UP IMBALANCE NETTING OPTIMIZATION')

        ###SETS###
        self.set_TIME = m.timestamps_high_str
        self.set_INTERVALS = list(range(self.set_TIME.__len__() - 1))
        self.set_AREAS = m.areas
        self.set_ACLINES = m.ac_unidir
        self.set_ACINDX = []
        for a in self.set_ACLINES:
            self.set_ACINDX.append(a[0])

        ###PARAMETERS###
        self.param_IMBALANCES = m.imbalances
        self.param_ATC = m.atc_high
        self.param_ALPHA = 10**(-5)
        self.param_BETA = 10**(-8)

        ###VARIABLES###
        self.var_AC = self.gm.addVars(self.set_ACINDX, self.set_TIME, name='AC', vtype=GRB.CONTINUOUS, lb=0, ub=float('inf'))
        # self.var_ACDIFFPOS = self.gm.addVars(self.set_ACINDX, self.set_INTERVALS, name='ACDIFFPOS', vtype=GRB.CONTINUOUS, lb=0, ub=float('inf'))
        # self.var_ACDIFFNEG = self.gm.addVars(self.set_ACINDX, self.set_INTERVALS, name='ACDIFFNEG', vtype=GRB.CONTINUOUS, lb=0, ub=float('inf'))
        self.var_IMBDIFFPOS = self.gm.addVars(self.set_AREAS, self.set_INTERVALS, name='IMBDIFFPOS', vtype=GRB.CONTINUOUS, lb=0, ub=float('inf'))
        self.var_IMBDIFFNEG = self.gm.addVars(self.set_AREAS, self.set_INTERVALS, name='IMBDIFFNEG', vtype=GRB.CONTINUOUS, lb=0, ub=float('inf'))
        self.var_BALANCINGUP = self.gm.addVars(self.set_AREAS, self.set_TIME, name='BALANCINGUP', vtype=GRB.CONTINUOUS, lb=0, ub=float('inf'))
        self.var_BALANCINGDOWN = self.gm.addVars(self.set_AREAS, self.set_TIME, name='BALANCINGDOWN', vtype=GRB.CONTINUOUS, lb=0, ub=float('inf'))

        ###OPTIMIZATION PROBLEM###
        self.setup_objective()
        self.setup_transmlimits()
        # self.setup_transmdiffpos()
        # self.setup_transmdiffneg()
        self.setup_imbdiffpos()
        self.setup_imbdiffneg()
        self.setup_balance()
        self.gm.update()

    def setup_objective(self):
        print('SETTING UP OBJECTIVE FUNCTION')
        obj = gp.LinExpr()
        obj.addTerms([1] * len(self.set_TIME) * len(self.set_AREAS),
                     [self.var_BALANCINGUP[a, t] for a in self.set_AREAS for t in self.set_TIME])
        obj.addTerms([1] * len(self.set_TIME) * len(self.set_AREAS),
                     [self.var_BALANCINGDOWN[a, t] for a in self.set_AREAS for t in self.set_TIME])
        obj.addTerms([self.param_ALPHA] * len(self.set_ACINDX) * len(self.set_TIME),
                     [self.var_AC[l, t] for l in self.set_ACINDX for t in self.set_TIME])
        obj.addTerms([self.param_BETA] * len(self.set_AREAS) * len(self.set_INTERVALS),
                     [self.var_IMBDIFFPOS[a, t] for a in self.set_AREAS for t in self.set_INTERVALS])
        obj.addTerms([self.param_BETA] * len(self.set_AREAS) * len(self.set_INTERVALS),
                     [self.var_IMBDIFFNEG[a, t] for a in self.set_AREAS for t in self.set_INTERVALS])
        self.gm.setObjective(obj, sense=GRB.MINIMIZE)

    def setup_transmlimits(self):
        print('SETTING UP TRANSMISSION CONSTRAINT')
        self.constr_ACMAX = {}
        for l in self.set_ACINDX:
            for t in self.set_TIME:
                left_hand = self.var_AC[l, t]
                right_hand = self.param_ATC[l][t]
                self.constr_ACMAX[l, t] = self.gm.addLConstr(
                    lhs = left_hand,
                    sense = GRB.LESS_EQUAL,
                    rhs = right_hand,
                    name = f'ACMAX[{l, t}]'
                )

    # def setup_transmdiffpos(self):
    #     print('SETTING UP TRANSMISSION DIFF CONSTRAINT 1')
    #     self.constr_ACDIFFPOS = {}
    #     for l in self.set_ACINDX:
    #         for i in self.set_INTERVALS:
    #             left_hand = self.var_AC[l, self.set_TIME[i + 1]] - self.var_AC[l, self.set_TIME[i]]
    #             right_hand = self.var_ACDIFFPOS[l, i]
    #             self.constr_ACDIFFPOS[l, i] = self.gm.addLConstr(
    #                 lhs = left_hand,
    #                 sense = GRB.LESS_EQUAL,
    #                 rhs = right_hand,
    #                 name = f'ACDIFFPOS[{l, i}]'
    #             )
    #
    # def setup_transmdiffneg(self):
    #     print('SETTING UP TRANSMISSION DIFF CONSTRAINT 2')
    #     self.constr_ACDIFFNEG = {}
    #     for l in self.set_ACINDX:
    #         for i in self.set_INTERVALS:
    #             left_hand = self.var_AC[l, self.set_TIME[i]] - self.var_AC[l, self.set_TIME[i + 1]]
    #             right_hand = self.var_ACDIFFNEG[l, i]
    #             self.constr_ACDIFFNEG[l, i] = self.gm.addLConstr(
    #                 lhs = left_hand,
    #                 sense = GRB.LESS_EQUAL,
    #                 rhs = right_hand,
    #                 name = f'ACDIFFNEG[{l, i}]'
    #             )

    def setup_imbdiffpos(self):
        print('SETTING UP IMBALANCE DIFF CONSTRAINT 1')
        self.constr_IMBDIFFPOS = {}
        for a in self.set_AREAS:
            for i in self.set_INTERVALS:
                left_hand = (self.var_BALANCINGUP[a, self.set_TIME[i + 1]] - self.var_BALANCINGDOWN[a, self.set_TIME[i + 1]]) - \
                (self.var_BALANCINGUP[a, self.set_TIME[i]] - self.var_BALANCINGDOWN[a, self.set_TIME[i]])
                right_hand = self.var_IMBDIFFPOS[a, i]
                self.constr_IMBDIFFPOS[a, i] = self.gm.addLConstr(
                    lhs = left_hand,
                    sense = GRB.LESS_EQUAL,
                    rhs = right_hand,
                    name = f'IMBDIFFPOS[{a, i}]'
                )

    def setup_imbdiffneg(self):
        print('SETTING UP IMBALANCE DIFF CONSTRAINT 2')
        self.constr_IMBDIFFNEG = {}
        for a in self.set_AREAS:
            for i in self.set_INTERVALS:
                left_hand = (self.var_BALANCINGUP[a, self.set_TIME[i]] - self.var_BALANCINGDOWN[a, self.set_TIME[i]]) - \
                (self.var_BALANCINGUP[a, self.set_TIME[i + 1]] - self.var_BALANCINGDOWN[a, self.set_TIME[i + 1]])
                right_hand = self.var_IMBDIFFNEG[a, i]
                self.constr_IMBDIFFNEG[a, i] = self.gm.addLConstr(
                    lhs = left_hand,
                    sense = GRB.LESS_EQUAL,
                    rhs = right_hand,
                    name = f'IMBDIFFNEG[{a, i}]'
                )


    def setup_balance(self):
        print('SETTING UP POWER BALANCE CONSTRAINT')
        self.constr_balance = {}
        for a in self.set_AREAS:
            for t in self.set_TIME:
                left_hand = self.param_IMBALANCES[a][t] + self.var_BALANCINGUP[a, t] - self.var_BALANCINGDOWN[a, t]
                for l in self.set_ACLINES:
                    if l[1] == a:
                        left_hand += self.var_AC[l[0], t]
                    elif l[2] == a:
                        left_hand -= self.var_AC[l[0], t]
                right_hand = 0
                self.constr_balance[a, t] = self.gm.addLConstr(
                    lhs = left_hand,
                    sense = GRB.EQUAL,
                    rhs = right_hand,
                    name = f'BALANCE[{a, t}]'
                )

class GurobiModel_Alt:

    def __init__(self, name, interval):
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            self.gm = gp.Model(name, env=env)
        #self.gm = gp.Model(name)
        self.start = interval[0]
        self.end = interval[-1] + 1
        self.name = name

    def setup_problem(self, m):
        """ PM = parent model """
        self.pm = m

        ###SETS###
        self.set_TIME = m.timestamps_high_str[self.start: self.end]
        self.set_INTERVALS = list(range(self.set_TIME.__len__() - 1))
        self.set_AREAS = m.areas
        self.set_ACLINES = m.ac_unidir
        self.set_ACINDX = []
        for a in self.set_ACLINES:
            self.set_ACINDX.append(a[0])

        ###PARAMETERS###
        self.param_IMBALANCES = m.imbalances.loc[self.set_TIME]
        self.param_ATC = m.atc_high.loc[self.set_TIME]
        self.param_ALPHA = 10**(-5)
        self.param_BETA = 10**(-8)

        ###VARIABLES###
        self.var_AC = self.gm.addVars(self.set_ACINDX, self.set_TIME, name='AC', vtype=GRB.CONTINUOUS, lb=0, ub=float('inf'))
        self.var_IMBDIFFPOS = self.gm.addVars(self.set_AREAS, self.set_INTERVALS, name='IMBDIFFPOS', vtype=GRB.CONTINUOUS, lb=0, ub=float('inf'))
        self.var_IMBDIFFNEG = self.gm.addVars(self.set_AREAS, self.set_INTERVALS, name='IMBDIFFNEG', vtype=GRB.CONTINUOUS, lb=0, ub=float('inf'))
        self.var_BALANCINGUP = self.gm.addVars(self.set_AREAS, self.set_TIME, name='BALANCINGUP', vtype=GRB.CONTINUOUS, lb=0, ub=float('inf'))
        self.var_BALANCINGDOWN = self.gm.addVars(self.set_AREAS, self.set_TIME, name='BALANCINGDOWN', vtype=GRB.CONTINUOUS, lb=0, ub=float('inf'))

        ###OPTIMIZATION PROBLEM###
        self.setup_objective()
        self.setup_transmlimits()
        self.setup_imbdiffpos()
        self.setup_imbdiffneg()
        self.setup_balance()
        self.gm.update()

    def setup_objective(self):
        print('SETTING UP OBJECTIVE FUNCTION')
        obj = gp.LinExpr()
        obj.addTerms([1] * len(self.set_TIME) * len(self.set_AREAS),
                     [self.var_BALANCINGUP[a, t] for a in self.set_AREAS for t in self.set_TIME])
        obj.addTerms([1] * len(self.set_TIME) * len(self.set_AREAS),
                     [self.var_BALANCINGDOWN[a, t] for a in self.set_AREAS for t in self.set_TIME])
        obj.addTerms([self.param_ALPHA] * len(self.set_ACINDX) * len(self.set_TIME),
                     [self.var_AC[l, t] for l in self.set_ACINDX for t in self.set_TIME])
        obj.addTerms([self.param_BETA] * len(self.set_AREAS) * len(self.set_INTERVALS),
                     [self.var_IMBDIFFPOS[a, t] for a in self.set_AREAS for t in self.set_INTERVALS])
        obj.addTerms([self.param_BETA] * len(self.set_AREAS) * len(self.set_INTERVALS),
                     [self.var_IMBDIFFNEG[a, t] for a in self.set_AREAS for t in self.set_INTERVALS])
        self.gm.setObjective(obj, sense=GRB.MINIMIZE)

    def setup_transmlimits(self):
        print('SETTING UP TRANSMISSION CONSTRAINT')
        self.constr_ACMAX = {}
        for l in self.set_ACINDX:
            for t in self.set_TIME:
                left_hand = self.var_AC[l, t]
                right_hand = self.param_ATC[l][t]
                self.constr_ACMAX[l, t] = self.gm.addLConstr(
                    lhs = left_hand,
                    sense = GRB.LESS_EQUAL,
                    rhs = right_hand,
                    name = f'ACMAX[{l, t}]'
                )

    def setup_imbdiffpos(self):
        print('SETTING UP IMBALANCE DIFF CONSTRAINT 1')
        self.constr_IMBDIFFPOS = {}
        for a in self.set_AREAS:
            for i in self.set_INTERVALS:
                left_hand = (self.var_BALANCINGUP[a, self.set_TIME[i + 1]] - self.var_BALANCINGDOWN[a, self.set_TIME[i + 1]]) - \
                (self.var_BALANCINGUP[a, self.set_TIME[i]] - self.var_BALANCINGDOWN[a, self.set_TIME[i]])
                right_hand = self.var_IMBDIFFPOS[a, i]
                self.constr_IMBDIFFPOS[a, i] = self.gm.addLConstr(
                    lhs = left_hand,
                    sense = GRB.LESS_EQUAL,
                    rhs = right_hand,
                    name = f'IMBDIFFPOS[{a, i}]'
                )

    def setup_imbdiffneg(self):
        print('SETTING UP IMBALANCE DIFF CONSTRAINT 2')
        self.constr_IMBDIFFNEG = {}
        for a in self.set_AREAS:
            for i in self.set_INTERVALS:
                left_hand = (self.var_BALANCINGUP[a, self.set_TIME[i]] - self.var_BALANCINGDOWN[a, self.set_TIME[i]]) - \
                (self.var_BALANCINGUP[a, self.set_TIME[i + 1]] - self.var_BALANCINGDOWN[a, self.set_TIME[i + 1]])
                right_hand = self.var_IMBDIFFNEG[a, i]
                self.constr_IMBDIFFNEG[a, i] = self.gm.addLConstr(
                    lhs = left_hand,
                    sense = GRB.LESS_EQUAL,
                    rhs = right_hand,
                    name = f'IMBDIFFNEG[{a, i}]'
                )


    def setup_balance(self):
        print('SETTING UP POWER BALANCE CONSTRAINT')
        self.constr_balance = {}
        for a in self.set_AREAS:
            for t in self.set_TIME:
                left_hand = self.param_IMBALANCES[a][t] + self.var_BALANCINGUP[a, t] - self.var_BALANCINGDOWN[a, t]
                for l in self.set_ACLINES:
                    if l[1] == a:
                        left_hand += self.var_AC[l[0], t]
                    elif l[2] == a:
                        left_hand -= self.var_AC[l[0], t]
                right_hand = 0
                self.constr_balance[a, t] = self.gm.addLConstr(
                    lhs = left_hand,
                    sense = GRB.EQUAL,
                    rhs = right_hand,
                    name = f'BALANCE[{a, t}]'
                )









