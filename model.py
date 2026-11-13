"""

COMAP Problem A 1988

Ian Morton, Ethan Wilson, Mateo Corrales Enriquez

"""

from __future__ import division # safety with double division
from pyomo.environ import *
from pyomo.opt import SolverFactory
from sys import argv
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Optional

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
M.crate_min = Param(M.crate, within=NonNegativeIntegers)

# ==============
# Variables
# ==============
M.num_crates = Var(M.car, M.crate, within=NonNegativeIntegers)

# ==============
# Objective
# ==============
def CalcUsedSpace(M):
   return sum (M.num_crates[car,crate]*M.thickness[crate] for crate in M.crate for car in M.car)
M.UsedSpace = Objective(rule=CalcUsedSpace, sense=maximize)

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

# def EnsureMinCrates(M,crate):
#    return sum(M.num_crates[car,crate] for car in M.car) >= M.crate_min[crate]
# M.MinCrates = Constraint(M.crate, rule=EnsureMinCrates)

# ============================
# Create a problem instance
# ============================
def main(argv: Optional[List[str]] = None) -> None:

   parser = ArgumentParser(
      description="Calculate a solution to the Railroad Flatcar problem."
   )

   parser.add_argument(
      "-f", "--file",
      type=Path,
      default=Path("small.dat"),
      help="Path to the input .dat file (default: small.dat)."
   )

   args = parser.parse_args(argv)

   # Resolve and validate the input file
   data_path = args.file.resolve()

   if not data_path.is_file():
      parser.error(f"Input file does not exist: {data_path}")

   print(f"Using input file: {data_path}")

   # Create the concrete model instance
   instance = M.create_instance(str(data_path))

   # Initialize the solver
   opt = SolverFactory("gurobi")

   # Solve the model
   soln = opt.solve(instance)

   # Load the solution into the model
   instance.solutions.load_from(soln)

   # Print the results
   print(
      "Termination Condition was "
      + str(soln.solver.termination_condition)
   )

   if soln.solver.termination_condition == TerminationCondition.optimal:
      instance.solutions.load_from(soln)
      instance.display()
   else:
      print("Solver did not find an optimal solution.")


if __name__ == "__main__":
   main()