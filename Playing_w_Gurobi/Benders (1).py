from gurobipy import *
import numpy as np

##### Building the Data #####

# Read the data from file
import BendersData as Data

CutViolationTolerance = 0.0001
OptionForSubproblem = 1 # 0 ==> Use Primal, 1==> Use Dual

f = Data.f            # facility opening costs (j)
c = Data.c            # transportation costs (i,j)
Lambda = Data.Lambda  # penalty costs (j)
u = Data.u            # facility capacities (i)
d = Data.d            # demands (s,j)

nS = len(d)   # the number of scenarios
p = [1.0/nS] * nS         # scenario probabilities (assuming equally likely scenarios)

# Build sets
I = range(len(c))
J = range(len(Lambda))
S = range(nS)

def ModifyAndSolveSP(s):
    # Modify constraint right-hand side (RHS) for scenario s
    for i in I:
        CapacityConsts[i].rhs = u[i] * xsol[i]
    for j in J:
        DemandConsts[j].rhs = d[s][j]
        SP.update()

    # Solve primal subproblem and extract dual solution
    SP.optimize()

    # Extract dual variables (shadow prices) from constraints
    pi_sol = [DemandConsts[j].Pi for j in J]
    gamma_sol = [CapacityConsts[i].Pi for i in I]

    SPobj = SP.objVal

    print("Subproblem " + str(s))
    print('SPobj: %g' % SPobj)
    print("pi_sol: " + str(pi_sol))
    print("gamma_sol: " + str(gamma_sol))

    # Check whether a violated Benders cut is found
    # Benders cut is violated if lower bound n[s] < actual objective value SPobj
    CutFound = False
    if(nsol[s] < SPobj - CutViolationTolerance): # Found Benders cut is violated at the current master solution
        CutFound = True

    return SPobj, CutFound, pi_sol, gamma_sol

def ModifyAndSolveDSP(s):
    # Modify dual subproblem objective coefficients for scenario s
    for i in I:
        gamma[i].obj = u[i] * xsol[i]
    for j in J:
        pi[j].obj = d[s][j]

    DSP.update()

    # Solve dual subproblem
    DSP.optimize()

    # Extract primal solution from dual subproblem (dual-dual = primal)
    pi_sol = [pi[j].x for j in J]
    gamma_sol = [gamma[i].x for i in I]

    DSPobj = DSP.objVal

    print("Subproblem " + str(s))
    print('DSPobj: %g' % DSPobj)
    print("pi_sol: " + str(pi_sol))
    print("gamma_sol: " + str(gamma_sol))

    # Check whether a violated Benders cut is found
    # Benders cut is violated if lower bound n[s] < actual objective value DSPobj
    CutFound = False
    if(nsol[s] < DSPobj - CutViolationTolerance): # Found Benders cut is violated at the current master solution
        CutFound = True

    return DSPobj, CutFound, pi_sol, gamma_sol

##### Build the Master Problem #####
# Master problem: minimize first-stage decisions (facility opening) with recourse approximation
MP = Model("MP")
MP.Params.outputFlag = 0  # turn off output
MP.Params.method = 1      # dual simplex

# First-stage variables: facility opening decisions (binary)
x = MP.addVars(I, vtype=GRB.BINARY, obj=f, name='x')

# Second-stage recourse variables: lower bounds on expected recourse cost per scenario
# n[s] is the approximation of second-stage cost for scenario s
n = MP.addVars(S, obj=p, name='n')
# Note: Default variable bounds are LB = 0 and UB = infinity
# n[s] coefficients are p[s] (scenario probabilities)

MP.modelSense = GRB.MINIMIZE

##### Build the Subproblem(s) #####
if OptionForSubproblem == 0:
    # Build Primal Subproblem (SP)
    # Primal problem: minimize transportation and penalty costs given facility opening decisions
    SP = Model("SP")
    
    # Second-stage variables:
    y = SP.addVars(I, J, obj=c, name='y')  # transportation variables: y[i,j]
    z = SP.addVars(J, obj=Lambda, name='z')  # penalty variables for unmet demand: z[j]

    DemandConsts = []
    CapacityConsts = []
    
    # Demand constraints: supply + penalty = demand for each customer j
    for j in J:
        DemandConsts.append(SP.addConstr((y.sum('*',j) + z[j] == 0), "Demand"+str(j)))
    
    # Production constraints: supply cannot exceed available capacity at facility i
    for i in I:
        CapacityConsts.append(SP.addConstr((y.sum(i,'*') <= 0), "Capacity" + str(i)))
    # NOTE: RHS of constraints will be set inside the loop later

    SP.modelSense = GRB.MINIMIZE
    SP.Params.outputFlag = 0

else:
    # Build Dual Subproblem (DSP)
    # Dual problem: maximize demand times dual variables subject to dual constraints
    DSP = Model("DSP")
    
    # Dual variables for demand constraints: pi[j] (bounded above by penalty cost Lambda)
    pi = DSP.addVars(J, ub=Lambda, obj=d[1], name='pi')
    
    # Dual variables for capacity constraints: gamma[i] (non-positive)
    gamma = []
    for i in I:
        gamma.append(DSP.addVar(lb = -GRB.INFINITY, ub = 0, obj=u[i], name="gamma"+str(i)))

    # Dual constraints: complementary to primal y[i,j] variables
    # pi[j] + gamma[i] <= c[i,j] (transportation cost)
    DSP.addConstrs((pi[j] + gamma[i] <= c[i][j] for i in I for j in J), name='For_y')
    # Note: Dual constraints corresponding to z variables are implicit (UBs on pi vars)

    DSP.modelSense = GRB.MAXIMIZE
    DSP.Params.outputFlag = 0

##### Benders Decomposition Loop #####
# Iteratively solve master problem and generate cutting planes from subproblems
CutFound = True
NoIters = 0
BestUB = GRB.INFINITY

while(CutFound):
    NoIters += 1
    CutFound = False

    # Solve Master Problem
    MP.update()
    MP.optimize()

    # Get Master Problem solution
    MPobj = MP.objVal
    print('MPobj: %g' % MPobj)

    # Extract first-stage decision: which facilities to open (x[i] = 1 or 0)
    xsol = [0 for i in I]
    for i in I:
        if x[i].x > 0.99:
            xsol[i] = 1

    # Extract second-stage recourse lower bounds
    nsol = [n[s].x for s in S]
    print("xsol: " + str(xsol))
    print("nsol: " + str(nsol))

    # Compute upper bound: first-stage cost + expected second-stage cost
    UB = np.dot(f,xsol)

    # Solve subproblems for each scenario and generate cuts
    for s in S:
        if OptionForSubproblem == 0:
            # Use primal subproblem and extract dual solution
            Qvalue, CutFound_s, pi_sol, gamma_sol = ModifyAndSolveSP(s)
        else:
            # Use dual subproblem (equivalent to primal)
            Qvalue, CutFound_s, pi_sol, gamma_sol = ModifyAndSolveDSP(s)

        # Add scenario cost to upper bound
        UB += p[s] * Qvalue

        # If violated cut found, add Benders cut to master problem
        if(CutFound_s):
            CutFound = True
            # Benders cut: n[s] >= d[s]'*pi_sol + u*gamma_sol'*x
            # Rearranged: n[s] - d[s]'*pi_sol - u*gamma_sol'*x >= 0
            expr = LinExpr(n[s] - quicksum(d[s][j]*pi_sol[j] for j in J) - quicksum(u[i]*gamma_sol[i]*x[i] for i in I))
            MP.addConstr(expr >= 0)
            print("CUT: " + str(expr) + " >= 0")

    # Track best upper bound found
    if(UB < BestUB):
        BestUB = UB
    print("UB: " + str(UB) + "\n")
    print("BestUB: " + str(BestUB) + "\n")

print('\nOptimal Solution:')
print('MPobj: %g' % MPobj)
print("xsol: " + str(xsol))
print("nsol: " + str(nsol))
print("NoIters: " + str(NoIters))