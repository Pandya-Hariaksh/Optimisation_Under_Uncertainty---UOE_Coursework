from gurobipy import *
import numpy as np

##### Building the Data #####

# Read the data from file
import FL_Data as Data

f = Data.f            # facility opening costs (j)
c = Data.c            # transportation costs (i,j)
Lambda = Data.Lambda  # penalty costs (j)
u = Data.u            # facility capacities (i)
d = Data.d            # demands (s,j)

nS = len(d)           # the number of scenarios
p = [1.0/nS] * nS     # scenario probabilities (assuming equally likely scenarios)

# Build sets
I = range(len(c))
J = range(len(Lambda))
S = range(nS)

# Build second-stage objective coefficients: Note that we scale them with scenario probabilities
Lambda_ForAllScen = [p[s] * Lambda[j] for j in J for s in S]
c_ForAllScen = [p[s] * c[i][j] for i in I for j in J for s in S]

##### Start building the Model #####
# Extensive form: Full stochastic program with all scenarios modeled explicitly
# Combines first-stage decisions (facility opening) with all scenario-specific recourse
m = Model("2SP_ExtForm")

# First-stage variables: facility opening decisions (binary)
# Decision made here-and-now before demand is realized
x = m.addVars(I, vtype=GRB.BINARY, obj=f, name='x')

# Second-stage variables: transportation and unmet demand decisions (scenario-dependent)
# Recourse decisions made after demand is revealed (separate decision for each scenario)
y = m.addVars(I, J, S, obj=c_ForAllScen, name='y')
z = m.addVars(J, S, obj=Lambda_ForAllScen, name='z')

# Note: Default variable bounds are LB = 0 and UB = infinity
# y[i,j,s] = units transported from facility i to customer j in scenario s
# z[j,s] = unmet demand (shortage) for customer j in scenario s

m.modelSense = GRB.MINIMIZE

# Demand constraints: supply + shortage = demand for each customer j and scenario s
# Constraint: y[*,j,s] + z[j,s] == d[s][j]
# Meaning: total shipments to customer j + unmet demand = demand in scenario s
m.addConstrs(
  (y.sum('*',j,s) + z[j,s] == d[s][j] for j in J for s in S), name='Demand');

# Capacity constraints: cannot exceed facility i's capacity (if open) in scenario s
# Constraint: y[i,*,s] <= u[i]*x[i]
# Meaning: total shipments from facility i <= capacity * (1 if open, 0 if closed)
m.addConstrs(
  (y.sum(i,'*',s) <= u[i]*x[i] for i in I for s in S), name='Capacity');

##### Solve the extensive form #####
# Solves two-stage stochastic program with explicit scenario enumeration
m.optimize()

OptimalValue_2SP = m.objVal
print('\nEXPECTED COST (Optimal 2-Stage Solution): %g' % OptimalValue_2SP)

print('\nOPTIMAL FACILITY CONFIGURATION:')
for i in I:
    if x[i].x > 0.99:
        print('Facility %s is open' % i)
        for j in J:
            if y[i,j,0].x > 0.00001:
                print('  Transport %g units to customer %s in scenario 0' % \
                      (y[i,j,0].x, j))

print('\nAVERAGE UNMET DEMAND (across all scenarios):')
for j in J:
    AvgUnmet = sum(z[j,s].x for s in S)/nS
    print('   Customer %s: %g units' % (j, AvgUnmet))

############ Build and solve the mean value problem ############
# Mean Value Problem (MVP): Deterministic approximation using expected demand
# Ignores demand uncertainty and uses average (expected) demand only
# Provides comparison point to evaluate cost of stochastic solution

# Compute average demands across all scenarios
d = np.array(d)
d_MV = np.sum(d,axis = 0)/nS

m_MV = Model("MVmodel")

# First-stage variables: facility opening decisions (binary)
# Same here-and-now decision as stochastic model
x_MV = m_MV.addVars(I, vtype=GRB.BINARY, obj=f, name='x_MV')

# Second-stage variables: transportation and unmet demand decisions (deterministic)
# Single recourse problem using expected demands (no scenario index)
# Note that this is a single scenario problem, so we don't use the scenario index
y_MV = m_MV.addVars(I, J, obj=c, name='y_MV')
z_MV = m_MV.addVars(J, obj=Lambda, name='z_MV')

m_MV.modelSense = GRB.MINIMIZE

# Demand constraints: supply + shortage = average demand
# Constraint: y[*,j] + z[j] == d_MV[j]
# Meaning: total shipments to customer j + unmet demand = expected demand
m_MV.addConstrs(
  (y_MV.sum('*',j) + z_MV[j] == d_MV[j] for j in J), 'Demand_MV');

# Capacity constraints: cannot exceed facility i's capacity (if open)
# Constraint: y[i,*] <= u[i]*x[i]
# Meaning: total shipments from facility i <= capacity * (1 if open, 0 if closed)
m_MV.addConstrs(
  (y_MV.sum(i,'*') <= u[i]*x_MV[i] for i in I), 'Capacity_MV');

# Solve mean value problem
m_MV.optimize()

print('\n\nMEAN VALUE SOLUTION (deterministic approximation):')
print('Expected cost of MVP: %g' % m_MV.objVal)
print('\nFacilities to open:')
for i in I:
    if x_MV[i].x > 0.99:
        print('Facility %s is open' % i)

############# Fix mean value solution and evaluate in stochastic model ##########
# Apply MVP facility configuration to stochastic model
# This shows how well the naive deterministic solution performs in reality

print('\n\nEVALUATING MEAN VALUE SOLUTION IN STOCHASTIC MODEL:')

# Fix first-stage variables to MVP solution
for i in I:
    if x_MV[i].x > 0.99:
        x[i].lb = 1.0  # Force facility to be open
    else:
        x[i].ub = 0.0  # Force facility to be closed

# Re-optimize second-stage recourse variables with fixed first-stage
m.update()
m.optimize()

OptimalValue_MVP_in_2SP = m.objVal
print('Expected cost of MVP solution under stochastic model: %g' % OptimalValue_MVP_in_2SP)

# Calculate Value of Stochastic Solution (VSS)
# VSS = Cost of MVP solution - Cost of optimal 2-stage solution
# VSS > 0 indicates benefit of considering uncertainty in first-stage decisions
VSS = OptimalValue_MVP_in_2SP - OptimalValue_2SP
print('\nValue of Stochastic Solution (VSS): %g' % VSS)
print('(Positive VSS indicates cost of ignoring uncertainty in first-stage decisions)')

# Calculate Expected Value of Perfect Information (EVPI)
# EVPI = Cost of optimal 2-stage solution - Cost of perfect information strategy
# (Not shown here, requires solving second stage with perfect demand knowledge)
print('\nInterpretation:')
print('- Stochastic solution considers all scenarios in facility decisions')
print('- Mean value solution uses only expected demand')
print('- VSS measures benefit of adapting facility strategy to demand uncertainty')