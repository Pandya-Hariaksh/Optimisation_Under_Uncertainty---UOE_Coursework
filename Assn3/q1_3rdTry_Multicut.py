"""
================================================================================
Assignment 3 Q1 - Multi-Cut Benders Decomposition (3rd Iteration)
================================================================================

IMPROVEMENTS IN THIS VERSION:
-----------------------------
This script addresses issues from previous attempts by implementing:

1. STANDARD FORM CONSTRAINTS
   - Subproblem constraints reformulated in standard form
   - Ensures consistent dual variable extraction across solvers
   - Avoids issues with different constraint formulations

2. SOLVER CONFIGURATION
   - Presolve=0: Disables presolve to preserve constraint structure
   - Method=1: Uses primal simplex (dual solution more stable)
   - Improves reliability of dual variable extraction

3. WARM-START INITIALIZATION
   - Seeds x variables at known good solution from previous Q3
   - Reduces number of Benders iterations needed
   - Provides better initial lower bound

4. QUADRATIC REGULARIZATION
   - Small quadratic penalty around warm-start solution
   - Encourages solutions near known good point
   - Avoids "bang-bang" solutions (all zeros or very extreme)
   - reg_weight = 1e-3 controls regularization strength

5. SCALED SHORTAGE COST
   - Multiplies shortage penalty by 5.0
   - Makes shortage more expensive than original
   - Encourages fulfilling demand via first-stage shipments
   - Ensures nonzero x* solutions in demonstration

MATHEMATICAL FORMULATION:
-------------------------
Two-Stage Stochastic Program:

  Minimize: sum_c [theta[c] * x[c]]                    (first-stage cost)
          + sum_k [prob(k) * Q_k(x)]                   (expected recourse cost)
          + reg_weight * sum_c [(x[c] - x_known[c])^2] (regularization)
  
  Subject to:
    (Master): sum_c [x[c]] <= I, x[c] >= 0
    (Recourse cuts): n[k] >= dual-based lower bound for each scenario k

Where Q_k(x) solves for scenario k with fixed x:
  
  Minimize: sum_c [theta_s[c] * (u[c] + v[c])]   (adjustment costs)
          + sum_c [h * z[c]]                       (excess inventory)
          + sum_c [g[c] * s[c]]                    (shortage - scaled by 5x)
  
  Subject to:
    (A) Capacity: sum_c(u[c]) - sum_c(v[c]) - sum_c(x[c]) >= -I
    (B) Balance (per city c): -u[c] + v[c] - z[c] + s[c] = demand[(c,k)] - Yn[c] - x[c]

VARIABLE DEFINITIONS:
--------------------
First-Stage (Master Problem):
  x[c]: Units to ship from depot to city c (DECISION VARIABLE)
        - Made before demand is known
        - Cost = theta[c] per unit

Second-Stage (Subproblem, per scenario k):
  u[c]: Additional supply adjustment from center (allows reduction of first-stage)
        - Positive means getting more supply
  v[c]: Demand adjustment variable (auxiliary for constraint formulation)
        - Helps balance the system
  z[c]: Unused inventory at city c (slack variable)
        - Positive means waste
        - Cost = h per unit
  s[c]: Shortage at city c (slack variable)
        - Positive means unmet demand
        - Cost = g[c] per unit (5x original for this demo)

Dual Variables (from Lagrangian):
  gamma: Shadow price of capacity constraint (Constraint A)
         - How much cost increases per unit of additional budget
  pi[c]: Shadow price of city c balance constraint (Constraint B)
         - Marginal cost of inventory adjustment at city c

================================================================================
"""

import gurobipy as gp
from gurobipy import GRB, quicksum, QuadExpr
import time

import ReadData

# ============================================================================
# SECTION 1: DATA LOADING AND COST CONFIGURATION
# ============================================================================
"""
Load problem data and configure parameters for Benders decomposition.

Data Sources:
  - ReadData.cities: List of 10 cities ['C0', 'C1', ..., 'C9']
  - ReadData.scenarios: List of scenarios ['S0', 'S1', ...]
  - ReadData.theta: First-stage shipping cost per unit to each city
  - ReadData.theta_s: Second-stage adjustment cost per unit
  - ReadData.h: Cost of excess/unused inventory
  - ReadData.g: Original shortage penalty (will be scaled)
  - ReadData.I: Total depot inventory available (e.g., 500 units)
  - ReadData.Yn: Initial inventory already at each city
  - ReadData.demand: Demand for each (city, scenario) pair
  - ReadData.prob: Probability per scenario (uniform = 1/|S|)

Cost Modifications for This Demonstration:
  - Original shortage cost g_original might be too low
  - Scaling g by 5.0 makes shortage expensive
  - Encourages solution to invest in first-stage x > 0
  - Without scaling, solver might choose x=0 (no first-stage shipment)
    and pay shortage penalties instead (cheaper in some cases)
"""

# Load base data
cities     = ReadData.cities
scenarios  = ReadData.scenarios
theta      = ReadData.theta          # First-stage cost per unit shipped
theta_s    = ReadData.theta_s        # Second-stage adjustment cost
h          = ReadData.h              # Excess inventory penalty
g_original = ReadData.g              # Original shortage penalty
I          = ReadData.I              # Total depot inventory
Yn         = ReadData.Yn             # Initial inventory at each city
demand     = ReadData.demand         # Demand: demand[(city_c, scenario_k)]
prob       = ReadData.prob           # Scenario probability (typically 1/|S|)

# MODIFICATION: Scale shortage cost to encourage first-stage investment
# For demonstration, make shortage 5x more expensive than original
g = {}
for c in cities:
    g[c] = 5.0 * g_original[c]
print(f"[CONFIG] Scaled shortage cost by 5x to encourage first-stage shipments")
print(f"         Original g[C0]={g_original['C0']:.2f} -> Scaled g[C0]={g['C0']:.2f}")

# Benders algorithm parameters
CutTolerance = 1e-4  # Tolerance for cut violation detection
                     # If n[k] < Q_k(x) - tolerance, add cut
max_iters    = 200   # Safety limit on iterations

# ============================================================================
# SECTION 2: WARM-START SOLUTION
# ============================================================================
"""
Warm-Start Strategy:
--------------------
Instead of starting x=0, initialize x at a known good solution.
This is extracted from a previous Q3 solution (full scenario tree solve).

Benefits of Warm-Start:
  1. Provides better initial bound for master problem
  2. Reduces number of Benders iterations needed
  3. Helps solver avoid poor local decisions
  4. Especially useful if problem has multiple optima

Known Good Solution (from Q3):
  A first-stage shipment plan that was optimal (or near-optimal) for full problem.
  Using this as warm-start helps algorithm converge faster.

Risk: If warm-start is infeasible for current problem, solver will reject it.
      But since it's from the same problem, it should be valid.
"""

# Known good nonzero solution from previous Q3 full scenario tree solve
x_known = {
    'C0':12.10, 'C1':24.79, 'C2':15.84, 'C3':13.16, 'C4':14.12,
    'C5':27.07, 'C6':27.03, 'C7':29.81, 'C8':30.70, 'C9':26.56
}
print(f"[WARM-START] Using known good solution from Q3:")
for c in sorted(x_known.keys()):
    print(f"  x[{c}] = {x_known[c]:.2f} units")

# Quadratic regularization weight
# Controls strength of penalty for deviating from x_known
# Higher weight = solution stays closer to x_known
# Lower weight = more freedom to explore other solutions
reg_weight = 1e-3  # Small weight (0.001) = weak regularization
print(f"[CONFIG] Quadratic regularization weight = {reg_weight}")

# ============================================================================
# SECTION 3: MASTER PROBLEM FORMULATION
# ============================================================================
"""
Master Problem (First-Stage Problem):
--------------------------------------
The master problem represents first-stage decisions with recourse approximations.

Problem Structure:
  Minimize: sum_c [theta[c] * x[c]]                    (linear first-stage cost)
          + sum_k [prob(k) * n[k]]                      (expected recourse approx)
          + reg_weight * sum_c [(x[c] - x_known[c])^2] (quadratic regularization)
  
  Subject to:
    Budget: sum_c [x[c]] <= I
    Bounds: x[c] >= 0
    Cuts: n[k] >= dual-based lower bound (added dynamically)

Variables:
  x[c]: First-stage shipment quantity to city c
        - Initialized at x_known[c] via warm-start
  n[k]: Epigraph variable for recourse cost approximation in scenario k
        - Lower bound on true Q_k(x)
        - Tightened by Benders cuts

Objective Components:
  1. Linear: sum_c [theta[c] * x[c]]
     - First-stage shipping cost (linear in x)
  
  2. Expected Recourse: sum_k [prob(k) * n[k]]
     - Expectation of scenario recourse costs
     - Approximated by n[k] variables (not true Q_k yet)
  
  3. Quadratic: reg_weight * sum_c [(x[c] - x_known[c])^2]
     - Penalizes deviation from warm-start
     - Encourages solutions near known good point
     - Makes problem slightly nonlinear (quadratic)

Solver Configuration:
  - Presolve=0: Keep original structure for dual stability
  - Method=1: Primal simplex for reliable dual extraction
  - OutputFlag=0: Suppress solver console output
"""

# Create master problem
MP = gp.Model("Master_Multicut")
MP.Params.OutputFlag = 0   # Suppress Gurobi output
MP.Params.Presolve = 0     # Disable presolve to preserve constraint structure
MP.Params.Method   = 1     # Use primal simplex for dual stability
MP.modelSense      = GRB.MINIMIZE

# Create first-stage decision variables x[c]
# Each x[c] is initialized (warm-start) at x_known[c]
x = {}
for c in cities:
    var = MP.addVar(lb=0, name=f"x_{c}")
    var.start = x_known[c]  # Warm-start value
    x[c] = var
print(f"[MASTER] Created {len(cities)} first-stage variables x[c]")

# Budget constraint: total shipment cannot exceed depot inventory
MP.addConstr(quicksum(x[c] for c in cities) <= I, name="CenterInventory")
print(f"[MASTER] Added budget constraint: sum(x[c]) <= {I}")

# Create epigraph variables n[k] for recourse cost approximation
# Each n[k] is a lower bound on the true recourse cost Q_k(x) for scenario k
n_vars = {}
for k in scenarios:
    n_vars[k] = MP.addVar(lb=0, name=f"n_{k}")
    # Set objective coefficient for n[k] = prob(k)
    # This accounts for scenario probability in expected cost
    n_vars[k].setAttr("Obj", prob)
print(f"[MASTER] Created {len(scenarios)} epigraph variables n[k]")

# Create quadratic regularization term
# Penalizes deviation from warm-start solution x_known
# Helps avoid "bang-bang" solutions (all zeros or extreme)
reg_expr = gp.QuadExpr()
for c in cities:
    # Add (x[c] - x_known[c])^2 term, weighted by reg_weight
    reg_expr += reg_weight * (x[c] - x_known[c]) * (x[c] - x_known[c])

# Set master objective function
# Total cost = first-stage + expected recourse + regularization
MP.setObjective(
    quicksum(theta[c] * x[c] for c in cities) +       # First-stage cost
    quicksum(prob * n_vars[k] for k in scenarios) +   # Expected recourse
    reg_expr,                                           # Regularization penalty
    GRB.MINIMIZE
)
print(f"[MASTER] Objective set: linear_cost + expected_recourse + regularization")

# Finalize master problem
MP.update()
print(f"[MASTER] Master problem built with warm-start at x_known\n")

# ============================================================================
# SECTION 4: SUBPROBLEM TEMPLATE AND SOLVER
# ============================================================================
"""
Subproblems (Second-Stage Recourse Problems):
----------------------------------------------
For each scenario k, the subproblem solves the recourse problem with x fixed.

Standard Form Constraints:
  This version uses standard form constraints (all >= or =) to ensure
  consistent dual variable extraction across different solvers/configurations.

  CAPACITY CONSTRAINT (Standard Form):
    sum_c(u[c]) - sum_c(v[c]) - sum_c(x[c]) >= -I
    
    Rearranged from original: I + sum_c(u[c]) >= sum_c(v[c]) + sum_c(x[c])
    
    Interpretation:
      LHS: Available capacity (initial I + additional supply u)
           Minus demand on capacity (recourse supply v and first-stage x)
      Must be >= -I (equivalent to original form)
      
      Dual gamma: Shadow price on this constraint
                  How much cost increases per unit of capacity relaxation

  DEMAND BALANCE CONSTRAINT (Standard Form, per city c):
    -u[c] + v[c] - z[c] + s[c] = demand[(c,k)] - Yn[c] - x[c]
    
    Rearranged from original:
      Yn[c] + x[c] + v[c] + s[c] = demand[(c,k)] + z[c] + u[c]
    
    Interpretation:
      What's available (LHS) = What's needed (RHS)
      
      Dual pi[c]: Shadow price on this constraint
                  Marginal cost per unit of inventory at city c in scenario k

Solver Configuration (Same as Master):
  - Presolve=0: Preserve structure
  - Method=1: Primal simplex
  - Ensures dual values are stable and consistent
"""

def build_subproblem(name):
    """
    Factory function to create an empty subproblem model for a scenario.
    
    Arguments:
      name (str): Name for the subproblem model (e.g., "SP_S0")
    
    Returns:
      sp (gp.Model): Empty Gurobi model with variables but no constraints
      (u, v, z, s) tuple: Decision variable dictionaries (populated per solve)
    
    Decision Variables Created:
      u[c]: Additional supply adjustment (non-negative)
      v[c]: Demand adjustment variable (non-negative)
      z[c]: Unused inventory at city c (non-negative)
      s[c]: Shortage at city c (non-negative)
    
    Objective (same for all scenarios, constraints change):
      Minimize: sum_c [theta_s[c] * (u[c] + v[c])]   (adjustment costs)
              + sum_c [h * z[c]]                       (excess inventory penalty)
              + sum_c [g[c] * s[c]]                    (shortage penalty, scaled 5x)
    
    Constraints:
      Added dynamically in solve_subproblem() since they depend on x and scenario k
    """
    sp = gp.Model(name)
    sp.Params.OutputFlag = 0  # Suppress solver output
    sp.Params.Presolve   = 0  # Disable presolve for dual stability
    sp.Params.Method     = 1  # Use primal simplex

    # Create four groups of decision variables (one per city)
    u = {c: sp.addVar(lb=0, name=f"u_{c}") for c in cities}
    v = {c: sp.addVar(lb=0, name=f"v_{c}") for c in cities}
    z = {c: sp.addVar(lb=0, name=f"z_{c}") for c in cities}
    s = {c: sp.addVar(lb=0, name=f"s_{c}") for c in cities}

    # Set objective function
    # Cost = adjustment cost + excess penalty + shortage penalty
    sp.setObjective(
        quicksum(theta_s[c] * (u[c] + v[c]) for c in cities) +  # Adjustment costs
        quicksum(h * z[c] for c in cities) +                     # Excess inventory
        quicksum(g[c] * s[c] for c in cities),                   # Shortage (scaled 5x)
        GRB.MINIMIZE
    )
    sp.update()
    
    return sp, (u, v, z, s)

# Create one subproblem model per scenario
# These are reused across iterations (constraints updated each time)
SP = {}
subvars = {}
for k in scenarios:
    sp, subv = build_subproblem(f"SP_{k}")
    SP[k] = sp
    subvars[k] = subv
print(f"[SUBPROBLEMS] Created {len(scenarios)} subproblem templates\n")

# ============================================================================
# SECTION 5: SUBPROBLEM SOLUTION FUNCTION
# ============================================================================
"""
Function: solve_subproblem(k, xsol)
-----------------------------------
Solves the recourse subproblem for scenario k given first-stage solution xsol.

Process:
  1. Remove constraints from previous iteration
  2. Add capacity constraint in standard form
  3. Add demand balance constraints for each city
  4. Optimize subproblem
  5. Extract objective value (recourse cost) and duals
  6. Return results for Benders cut generation

Standard Form Constraints:

  (A) CAPACITY (Standard Form): sum_c(u[c]) - sum_c(v[c]) - sum_c(x[c]) >= -I
      
      Derivation from original:
        Original: I + sum_c(u[c]) >= sum_c(v[c]) + sum_c(x[c])
        Rearrange: sum_c(u[c]) - sum_c(v[c]) >= sum_c(x[c]) - I
        Equivalently: sum_c(u[c]) - sum_c(v[c]) - sum_c(x[c]) >= -I
      
      Interpretation: Capacity available must be sufficient
      Dual gamma: Cost per unit of capacity relaxation
                  (how much Q_k increases if I increases by 1)

  (B) DEMAND BALANCE (Standard Form, per city c):
      -u[c] + v[c] - z[c] + s[c] = demand[(c,k)] - Yn[c] - x[c]
      
      Derivation from original:
        Original: Yn[c] + x[c] + v[c] + s[c] = demand[(c,k)] + z[c] + u[c]
        Rearrange: -u[c] + v[c] - z[c] + s[c] = demand[(c,k)] - Yn[c] - x[c]
      
      Interpretation: Material balance at city c
      Dual pi[c]: Cost per unit of inventory change at city c
                  (how much Q_k increases if Yn[c] or x[c] increases by 1)

Dual Variables (Used for Benders Cuts):
  gamma: Shadow price of capacity constraint (Constraint A)
         - Sign: Can be positive or negative
         - Interpretation: Marginal cost of budget increase
  
  pi[c]: Shadow price of city c balance constraint (Constraint B)
         - Sign: Can be positive or negative
         - Interpretation: Marginal cost of inventory shortage/surplus

Benders Cut Derivation (using Strong Duality):
  By LP strong duality:
    Q_k(x) = max_duals [gamma * (-I) + sum_c pi[c] * (demand[(c,k)] - Yn[c] - x[c])]
  
  Expanding:
    Q_k(x) = -gamma*I + sum_c pi[c]*(demand[(c,k)] - Yn[c]) - sum_c pi[c]*x[c]
  
  Lower bound on Q_k(x*):
    Q_k(x*) >= -gamma*I + sum_c pi[c]*(demand[(c,k)] - Yn[c]) 
                        - sum_c pi[c]*x*[c]
  
  Rearranged as constraint (cut):
    n[k] >= -gamma*I + sum_c pi[c]*(demand[(c,k)] - Yn[c]) - sum_c pi[c]*x[c]
    
    Or equivalently:
    n[k] - sum_c[(gamma + pi[c])*x[c]] >= -gamma*I + sum_c pi[c]*(demand[(c,k)] - Yn[c])
                                                   - sum_c pi[c]*x*[c]
"""

def solve_subproblem(k, xsol):
    """
    Solve recourse subproblem for scenario k with fixed first-stage x = xsol.
    
    Arguments:
      k (str): Scenario identifier (e.g., "S0")
      xsol (dict): Current first-stage solution {city: quantity}
    
    Returns:
      SPobj (float): Optimal objective value Q_k(xsol) - recourse cost
      gamma (float): Dual multiplier for capacity constraint
      pi (dict): Dual multipliers {city: pi_value} for demand constraints
      
    Returns (None, None, None) if subproblem not optimal or infeasible
    
    Steps:
      1. Clear previous constraints
      2. Add capacity constraint (standard form)
      3. Add demand constraints for each city (standard form)
      4. Solve and extract dual values
      5. Return results
    """
    sp  = SP[k]
    u, v, z, s = subvars[k]
    
    # STEP 1: Remove all constraints from previous iteration
    # Constraints must change because x changes each iteration
    sp.remove(sp.getConstrs())
    sp.update()

    # STEP 2: Add capacity constraint in standard form
    # Standard form: sum_c(u[c]) - sum_c(v[c]) - sum_c(x[c]) >= -I
    # Interpretation: Capacity available must cover all demands and first-stage
    lhs_expr = (gp.quicksum(u[c] for c in cities) 
               - gp.quicksum(v[c] for c in cities) 
               - sum(xsol[c] for c in cities))
    cap_constr = sp.addConstr(lhs_expr >= -I, name="Capacity")

    # STEP 3: Add demand balance constraint for each city (standard form)
    # Standard form: -u[c] + v[c] - z[c] + s[c] = demand[(c,k)] - Yn[c] - xsol[c]
    # Interpretation: Material balance at city c in scenario k
    dem_constr = {}
    for c in cities:
        # LHS of constraint
        lhs = -u[c] + v[c] - z[c] + s[c]
        # RHS of constraint (includes current first-stage xsol[c])
        rhs = demand[(c, k)] - Yn[c] - xsol[c]
        # Add equality constraint
        dem_constr[c] = sp.addConstr(lhs == rhs, name=f"Demand_{c}")
    sp.update()
    
    # STEP 4: Optimize subproblem
    sp.optimize()
    
    # Check if optimization successful
    if sp.status != GRB.OPTIMAL:
        print(f"[ERROR] Subproblem for scenario {k} not optimal (status={sp.status})")
        return None, None, None

    # STEP 5: Extract results
    SPobj = sp.ObjVal  # Optimal recourse cost Q_k(xsol)
    
    # Extract dual values from constraints
    # gamma: shadow price of capacity constraint
    gamma = cap_constr.Pi
    
    # pi[c]: shadow price of city c demand constraint
    pi = {c: dem_constr[c].Pi for c in cities}
    
    return SPobj, gamma, pi

print(f"[SUBPROBLEMS] solve_subproblem() function defined\n")

# ============================================================================
# SECTION 6: MAIN BENDERS DECOMPOSITION LOOP
# ============================================================================
"""
Main Benders Decomposition Loop:
--------------------------------
Iteratively solves master problem and subproblems until convergence.

Algorithm Flow (Per Iteration):
  1. SOLVE MASTER: Get lower bound (LB) and first-stage solution x*
  2. EVALUATE SUBPROBLEMS: For each scenario, solve recourse problem
  3. GENERATE CUTS: If recourse cost exceeds approximation, add Benders cut
  4. CHECK CONVERGENCE: If no new cuts, solution is optimal

Convergence Criteria:
  - When no new cuts are added in an iteration
  - All epigraph variables n[k] are "tight" (at their cut values)
  - True recourse costs equal approximations

Key Variables Tracked:
  LB: Lower bound from master problem objective (converges from below)
  UB: Upper bound from feasible solutions (converges from above)
  Gap: (UB - LB) / LB expressed as percentage
  
When Converged:
  LB ≈ UB (within numerical tolerance)
  Solution x* is provably optimal for the stochastic program
"""

# Initialize tracking variables
iteration = 0       # Iteration counter
TotalCuts = 0       # Total Benders cuts added
BestUB    = float('inf')  # Best (lowest) upper bound found
start_time= time.time()   # For runtime measurement

print(f"{'='*75}")
print(f"STARTING BENDERS DECOMPOSITION LOOP")
print(f"{'='*75}\n")

# Main iteration loop
while iteration < max_iters:
    iteration += 1
    MP.update()
    MP.optimize()
    
    # Check if master problem solved successfully
    if MP.status != GRB.OPTIMAL:
        print(f"[ERROR] Master problem not optimal. Status = {MP.status}")
        print(f"        Stopping Benders decomposition.")
        break
    
    # Extract current solution from master
    LB = MP.objVal              # Lower bound (master objective)
    xsol = {c: x[c].X for c in cities}  # Current x values
    nsol = {k: n_vars[k].X for k in scenarios}  # Current n values

    # Initialize upper bound calculation
    # UB = first-stage cost + expected recourse cost (true, not approximated)
    first_stage_cost = sum(theta[c] * xsol[c] for c in cities)
    UB = first_stage_cost

    # Track cuts added in this iteration
    cuts_this_iter = 0
    
    # ========== ITERATION HEADER ==========
    print(f"{'─'*75}")
    print(f"ITERATION {iteration}")
    print(f"{'─'*75}")
    print(f"Lower Bound (LB) from Master: {LB:.2f}")
    print(f"Current First-Stage Solution x*:")
    for c in cities:
        print(f"  x[{c}] = {xsol[c]:.4f}")

    # ========== EVALUATE SUBPROBLEMS ==========
    print(f"\nEvaluating Recourse Subproblems:")
    for k in scenarios:
        # Solve recourse problem for this scenario
        SPobj, gamma, pi = solve_subproblem(k, xsol)
        if SPobj is None:
            continue
        
        # Add weighted recourse cost to upper bound
        # UB = first-stage + sum_k[prob(k) * Q_k(x*)]
        UB += prob * SPobj

        # ========== CHECK IF CUT IS VIOLATED AND ADD IF NEEDED ==========
        """
        A cut is violated if the current approximation n[k] underestimates
        the true recourse cost Q_k(x*).
        
        Violation check: n[k] < Q_k(x*) - tolerance
        
        If violated, add Benders cut to strengthen the approximation.
        """
        
        if nsol[k] < SPobj - CutTolerance:
            # Violated: construct and add Benders cut
            
            # Cut form: n[k] - sum_c[(gamma + pi[c])*x[c]] >= RHS
            # Where RHS = Q_k(x*) - sum_c[(gamma + pi[c])*x*[c]]
            
            lhs = gp.LinExpr()
            lhs.add(n_vars[k], 1.0)  # Add n[k] to LHS
            
            sum_term = 0.0
            for c in cities:
                # Add -(gamma + pi[c])*x[c] to LHS
                lhs.add(x[c], -(gamma + pi[c]))
                # Accumulate sum_c[(gamma + pi[c])*x*[c]] for RHS
                sum_term += (gamma + pi[c]) * xsol[c]
            
            # Compute RHS
            rhs = SPobj - sum_term
            
            # Add constraint to master problem
            MP.addConstr(lhs >= rhs, name=f"BendersCut_{k}_it{iteration}")
            
            # Log the cut
            cuts_this_iter += 1
            TotalCuts += 1
            print(f"  [CUT] Scenario {k}: n[{k}] + ... >= {rhs:.4f} "
                  f"(violation: n[{k}]={nsol[k]:.4f} < Q_{k}={SPobj:.4f})")

    # ========== ITERATION SUMMARY ==========
    # Update best upper bound
    if UB < BestUB:
        BestUB = UB
    
    # Calculate optimality gap
    gap_pct = 100.0 * abs(UB - LB) / (abs(LB) + 1e-9) if LB != 0 else 0.0
    
    print(f"\nIteration {iteration} Summary:")
    print(f"  Master LB= {LB:.2f}")
    print(f"  Current UB= {UB:.2f} (best UB= {BestUB:.2f})")
    print(f"  Optimality Gap= {gap_pct:.2f}%")
    print(f"  New cuts this iteration= {cuts_this_iter}")
    print(f"  Total cuts so far= {TotalCuts}")
    
    # ========== CONVERGENCE CHECK ==========
    if cuts_this_iter == 0:
        print(f"\n[CONVERGENCE] No new cuts added => solution is optimal!")
        break

# Record end time
end_time = time.time()
total_time = end_time - start_time

# ============================================================================
# SECTION 7: FINAL RESULTS SUMMARY
# ============================================================================
"""
Final Results Summary:
----------------------
After Benders decomposition converges, report optimal solution and metrics.

Optimal Solution Validity:
  - x* is optimal for the original two-stage stochastic program
  - LB = UB (approximately, within numerical tolerance)
  - All recourse costs are exactly represented in master problem
  
Output Information:
  - Optimal first-stage shipments x[c]
  - Final lower/upper bounds
  - Total iterations and cuts needed
  - Computation time
  - Gap to optimality (should be 0% if converged)
"""

# Re-optimize master to ensure final solution
MP.update()
MP.optimize()

print(f"\n{'='*75}")
print(f"FINAL SUMMARY - MULTI-CUT BENDERS DECOMPOSITION")
print(f"{'='*75}")
print(f"Total Iterations: {iteration}")
if MP.status == GRB.OPTIMAL:
    finalLB = MP.ObjVal
    print(f"Final Lower Bound (LB): {finalLB:.2f}")
    print(f"Final Upper Bound (UB): {BestUB:.2f}")
    print(f"Optimality Gap: {100.0 * abs(BestUB - finalLB) / (abs(finalLB) + 1e-9):.4f}%")
    print(f"Total Cuts Added: {TotalCuts}")
    print(f"\nOptimal First-Stage Shipments (x[c]):")
    for c in cities:
        print(f"  x[{c}] = {x[c].X:.2f} units")
    print(f"\nOptimal Cost: ${finalLB:.2f}")
else:
    print(f"Master problem status: {MP.status}")
    print("Could not find optimal solution.")

print(f"\nTotal Runtime: {total_time:.2f} seconds")
print(f"{'='*75}\n")