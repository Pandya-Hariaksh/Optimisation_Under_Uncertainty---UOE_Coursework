import gurobipy as gp
from gurobipy import GRB

##### Example 1: Simple MIP Problem - Notes #####
"""
This is a basic binary integer programming problem.
Demonstrates fundamental Gurobi workflow: create model -> add variables -> set objective -> add constraints -> optimize

Problem:
  Maximize: x + y + 2*z
  Subject to:
    x + 2*y + 3*z <= 4
    x + y >= 1
    x, y, z ∈ {0, 1} (binary variables)

# Create a new model
m = gp.Model("mip1")
# Create variables
x = m.addVar(vtype=GRB.BINARY, name="x")
y = m.addVar(vtype=GRB.BINARY, name="y")
z = m.addVar(vtype=GRB.BINARY, name="z")

# Set objective: maximize x + y + 2*z
m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)
# Add constraints
m.addConstr(x + 2 * y + 3 * z <= 4, "c0")
m.addConstr(x + y >= 1, "c1")
# Optimize model
m.optimize()
for v in m.getVars():
    print('%s %g' % (v.VarName, v.X))
    print('Obj: %g' % m.ObjVal)
"""

##### Network Flow Problem: Multi-Commodity Flow #####
# Minimize cost to ship two commodities (Pencils, Pens) through a supply network
# Problem: Satisfy demand at customer nodes while respecting arc capacity constraints

##### Define Problem Data #####
# Two commodities to be shipped through the network
commodities = ['Pencils', 'Pens']

# Five nodes: two supply nodes (Detroit, Denver) and three demand nodes (Boston, New York, Seattle)
nodes = ['Detroit', 'Denver', 'Boston', 'New York', 'Seattle']

# Define arcs and their capacity constraints
# Arc: (origin, destination) with maximum flow capacity (units per arc, shared across commodities)
arcs, capacity = gp.multidict({
    ('Detroit','Boston'): 100,
    ('Detroit','New York'): 80,
    ('Detroit','Seattle'): 120,
    ('Denver','Boston'): 120,
    ('Denver','New York'): 120,
    ('Denver','Seattle'): 120 
})

# Transportation cost per unit for each commodity on each arc
# Cost: (commodity, origin, destination) with unit shipping cost
cost = {
    ('Pencils','Detroit','Boston'): 10,
    ('Pencils','Detroit','New York'): 20,
    ('Pencils','Detroit','Seattle'): 60,
    ('Pencils','Denver','Boston'): 40,
    ('Pencils','Denver','New York'): 40,
    ('Pencils','Denver','Seattle'): 30,
    ('Pens','Detroit','Boston'): 20,
    ('Pens','Detroit','New York'): 20,
    ('Pens','Detroit','Seattle'): 80,
    ('Pens','Denver','Boston'): 60,
    ('Pens','Denver','New York'): 70,
    ('Pens','Denver','Seattle'): 30 
}

# Node inflow/outflow: supply (+) and demand (-)
# Positive values: supply nodes (production available)
# Negative values: demand nodes (units required)
# Constraint: total supply = total demand (balanced network)
inflow = {
    ('Pencils','Boston'): -50,       # Demand: 50 units of Pencils
    ('Pencils','New York'): -50,     # Demand: 50 units of Pencils
    ('Pencils','Seattle'): -10,      # Demand: 10 units of Pencils
    ('Pencils','Denver'): 50,        # Supply: 50 units of Pencils available
    ('Pencils','Detroit'): 60,       # Supply: 60 units of Pencils available
    ('Pens','Boston'): -40,          # Demand: 40 units of Pens
    ('Pens','New York'): -30,        # Demand: 30 units of Pens
    ('Pens','Seattle'): -30,         # Demand: 30 units of Pens
    ('Pens','Denver'): 40,           # Supply: 40 units of Pens available
    ('Pens','Detroit'): 60           # Supply: 60 units of Pens available
}

##### Create Optimization Model #####
m = gp.Model("netflow")

##### Decision Variables #####
# flow[h,i,j] = units of commodity h shipped from node i to node j
# Variables indexed by commodity, origin, destination
# obj=cost: automatically incorporates cost coefficients into objective function
flow = m.addVars(commodities, arcs, obj=cost, name="flow")

##### Arc Capacity Constraints #####
# Total flow on arc (i,j) cannot exceed arc capacity (across all commodities)
# Constraint: sum_h flow[h,i,j] <= capacity[i,j]
# The '*' sums over all commodities
m.addConstrs(
    (flow.sum('*', i, j) <= capacity[i, j] for i, j in arcs), 
    "cap"
)

# Alternative formulation using explicit Python loop (Test):
# This approach iterates through each arc and adds constraints individually
# More verbose but arguably clearer about what is being constrained
"""
for i, j in arcs:
    # Add constraint for arc (i,j): all commodities' flows sum to capacity
    m.addConstr(
        sum(flow[h, i, j] for h in commodities) <= capacity[i, j], 
        "cap[%s, %s]" % (i, j)  # String formatting for constraint name
    )
"""

##### Flow Balance Constraints #####
# At each node (except sources/sinks), inflow equals outflow
# Conservation of flow: commodity cannot accumulate or disappear at nodes
# Constraint: sum_i flow[h,i,j] + inflow[h,j] == sum_k flow[h,j,k]
# where inflow[h,j] represents supply (+) or demand (-) at node j
m.addConstrs(
    (flow.sum(h, '*', j) + inflow[h, j] == flow.sum(h, j, '*') 
     for h in commodities for j in nodes), 
    "node"
)

# Flow balance meaning:
# - Left side: incoming flow + supply/demand at node
# - Right side: outgoing flow from node
# - Equation ensures: (inflow + supply) = (outflow - demand)

# Alternative formulation using quicksum and arc.select (commented out):
# This approach explicitly selects incoming and outgoing arcs
"""
m.addConstrs(
    (gp.quicksum(flow[h, i, j] for i, j in arcs.select('*', j)) +
     inflow[h, j] == gp.quicksum(flow[h, j, k] for j, k in arcs.select(j, '*'))
     for h in commodities for j in nodes), 
    "node"
)
"""

##### Solve the Optimization Problem #####
# Gurobi finds the minimum-cost flow satisfying all constraints
m.optimize()

##### Extract and Display Results #####
if m.Status == GRB.OPTIMAL:
    # Solution found: retrieve optimal flow values
    solution = m.getAttr("X", flow)
    
    # Print optimal flows for each commodity
    for h in commodities:
        print(f"\n{'='*50}")
        print(f"Optimal flows for {h}:")
        print(f"{'='*50}")
        
        for i, j in arcs:
            flow_amount = solution[h, i, j]
            
            # Only display routes with positive flow (ignore zero flows)
            if flow_amount > 0:
                cost_per_unit = cost[h, i, j]
                total_cost = flow_amount * cost_per_unit
                print(f"  {i:8s} -> {j:10s}: {flow_amount:6.0f} units  " +
                      f"(cost: ${cost_per_unit}/unit, total: ${total_cost:.0f})")
    
    # Summary statistics
    print(f"\n{'='*50}")
    print(f"Total Minimum Cost: ${m.ObjVal:.0f}")
    print(f"{'='*50}")
    
else:
    # No optimal solution found
    print(f"Optimization failed. Status: {m.Status}")
    if m.Status == GRB.INFEASIBLE:
        print("Model is infeasible - check supply/demand balance and arc capacities")
    elif m.Status == GRB.UNBOUNDED:
        print("Model is unbounded - check objective and constraints")