"""

COMAP Problem A 1988

Ian Morton, Ethan Wilson, Mateo Corrales Enriquez

"""

from __future__ import division # safety with double division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import os.path

# Instantiate and name the model
M = AbstractModel()
M.name = "Railroad Flatcars"

# Sets
M.crate = Set()
M.car = Set()
M.restricted = Set(within=M.crate)  # specially restricted crates

# Parameters
M.thickness = Param(M.crate, within=NonNegativeReals)
M.weight = Param(M.crate, within=NonNegativeReals)
M.crate_limit = Param(M.crate, within=NonNegativeReals)
M.car_length = Param(within=NonNegativeReals)
M.max_weight = Param(within=NonNegativeReals)
M.special_limit = Param(within=NonNegativeReals)

# ==============
# Variables
# ==============
M.num_crates = Var(M.car, M.crate, within=NonNegativeIntegers)

# ==============
# Objective
# ==============
def CalcWastedFloorSpace(M):
   return 2*M.car_length - sum (M.num_crates[car,crate]*M.thickness[crate] for crate in M.crate for car in M.car)
M.WastedFloorSpace = Objective(rule=CalcWastedFloorSpace, sense=minimize)

# ==============
# Constraints
# ==============
def EnsureCarLengthLimit(M,car):
   return sum (M.num_crates[car, crate]*M.thickness[crate] for crate in M.crate) <= M.car_length
M.CarLengthLimit = Constraint(M.car, rule=EnsureCarLengthLimit)

def EnsureCrateLimit(M,crate):
   return sum (M.num_crates[car, crate] for car in M.car) <= M.crate_limit[crate]
M.CrateLimit = Constraint(M.crate, rule=EnsureCrateLimit)

def EnsureWeightLimit(M,car):
   return sum (M.num_crates[car,crate]*M.weight[crate] for crate in M.crate) <= M.max_weight
M.WeightLimit = Constraint(M.car, rule=EnsureWeightLimit)

def EnsureSpecialRestriction(M):
   return sum(M.num_crates[car, crate]*M.thickness[crate] for car in M.car for crate in M.restricted) <= M.special_limit
M.SpecialRestriction = Constraint(rule=EnsureSpecialRestriction)


# ============================
# Create a problem instance
# ============================
this_dir = os.path.dirname(__file__)
data_path = os.path.join(this_dir, "model.dat")
instance = M.create_instance(data_path)
# instance.pprint()

# Indicate which solver to use
Opt = SolverFactory("gurobi")

# Generate a solution
Soln = Opt.solve(instance)
instance.solutions.load_from(Soln)

# Print the output
print("Termination Condition was "+str(Soln.Solver.Termination_condition))
display(instance)
# instance.pprint()