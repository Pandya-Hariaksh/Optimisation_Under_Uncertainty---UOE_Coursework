"""
Benders_Multicut_Q1_Refined.py
------------------------------

This script implements a multiple-cut Benders decomposition for Assignment 3, Q1 
using Gurobi. It follows these steps:

1. Load data (cities, scenarios, I, Yn, demand, theta, theta_s, h, g) from ReadData.py.

2. Define the recourse (second-stage) subproblem:
   For scenario k, given first-stage decision x, solve:
   
       Minimize: (sum over all cities n of) [theta_s[n] * y_n + h * leftover_n + g * shortage_n]
       
       Subject to:
           (sum over all cities n of) y_n  <=  I - (sum over all cities n of) x_n
           
           For each city n:
               Y_n + x_n + y_n = demand[(n,k)] + leftover_n - shortage_n
           
           y_n, leftover_n, shortage_n >= 0
   
   Physical Interpretation:
       - y_n: units of recourse supply sent to city n in second stage (after demand revealed)
       - leftover_n: excess inventory at city n (units not demanded)
       - shortage_n: unmet demand at city n (units in shortage)
       - Capacity: total recourse supply is limited by remaining budget I - (first-stage allocation)
       - Balance: initial inventory + first-stage + second-stage = demand + excess - shortage
       - Costs: penalize recourse supply (theta_s), excess inventory (h), and shortage (g)
   
   Its dual yields a valid Benders cut:
       eta_k >= mu * I + (sum over all cities n of) [pi_n * (demand[(n,k)] - Y_n)]
                - (sum over all cities n of) [(mu + pi_n) * x_n]
       
       where: mu = dual multiplier for capacity constraint (shadow price of recourse budget)
              pi_n = dual multiplier for city n balance constraint (shadow price of inventory)

3. Warm Start: Set x=0 (no first-stage allocation), solve each recourse subproblem 
   to get Q_k(0) (recourse cost with x=0), and force eta_k >= Q_k(0).

4. Benders Loop: In each iteration:
   a) Solve the master problem to get current lower bound and first-stage solution x*
   b) For each scenario k, solve the recourse subproblem at x* to get Q_k(x*) and duals
   c) Check if the current eta_k violates the Benders cut (is too small)
   d) If violated, add the cut: eta_k >= (dual-based affine function of x)
   e) Compute upper bound from feasible solution (first-stage + expected recourse)
   f) Check convergence: stop if no new cuts added or optimality gap below tolerance

5. Update the UB from feasible solutions and stop when convergence is achieved.
"""

import gurobipy as gp
from gurobipy import GRB
import time

# --------------------------
# 1. Data Loading
# --------------------------
# Load problem data from ReadData module
import ReadData
cities    = ReadData.cities         # List of city names, e.g., ['C0','C1',...,'C9']
scenarios = ReadData.scenarios      # List of scenario names, e.g., ['S0','S1',...]
prob      = 1.0 / len(scenarios)    # Probability per scenario (assume equally likely)
I         = ReadData.I              # Total distribution budget available in stage 1, e.g., 500.0
Yn        = ReadData.Yn             # Initial inventory at each city (given constant)
demand    = ReadData.demand         # Demand dictionary: demand[(city_n, scenario_k)]
theta1    = ReadData.theta          # First-stage cost per unit distributed
theta_s   = ReadData.theta_s        # Second-stage cost coefficient for recourse supply y_n
h         = ReadData.h              # Cost per unit of leftover (excess) inventory
g         = ReadData.g              # Cost per unit of shortage (unmet demand)

tol = 1e-6  # Tolerance for convergence (relative optimality gap)

# --------------------------
# 2. Recourse Subproblem Solver
# --------------------------
def solve_recourse_subproblem(k, xvals):
    """
    For scenario k and given first-stage solution xvals (a dictionary of x[n] values),
    solve the recourse (second-stage) subproblem.
    
    Recourse Problem Formulation:
    ==============================
    Minimize: (sum over all cities n of) [theta_s[n] * y_n + h * leftover_n + g * shortage_n]
    
    Subject to:
        Capacity Constraint:
            (sum over all cities n of) y_n <= I - (sum over all cities n of) xvals[n]
            
            Meaning: total recourse supply cannot exceed remaining budget after first-stage allocation
        
        Demand Balance Constraint (for each city n):
            Y_n + xvals[n] + y_n = demand[(n,k)] + leftover_n - shortage_n
            
            Meaning: starting inventory + first-stage supply + recourse supply
                   = actual demand + excess inventory - unmet demand
            
            Rearranged: Y_n + xvals[n] + y_n - leftover_n + shortage_n = demand[(n,k)]
        
        Non-negativity:
            y_n >= 0 (cannot have negative supply)
            leftover_n >= 0 (cannot have negative excess)
            shortage_n >= 0 (cannot have negative shortage)
    
    Solution Output:
        Q_val: optimal objective value (total recourse cost Q_k(x_values))
        mu: dual variable (Lagrange multiplier) for capacity constraint
            - Represents shadow price: increase in recourse cost per unit of additional budget
        pi[n]: dual variable for city n balance constraint
            - Represents shadow price: increase in recourse cost per unit of inventory at city n
    
    Benders Cut Generation:
        By strong duality in linear programming, the recourse cost satisfies:
        
        Q_k(x) = max_dual [mu * (I - sum_n x_n) + sum_n pi_n * (demand[(n,k)] - Y_n - x_n)]
        
        This implies for any x:
        Q_k(x) >= mu * I + sum_n [pi_n * (demand[(n,k)] - Y_n)] - sum_n [(mu + pi_n) * x_n]
        
        Therefore: eta_k >= mu * I + sum_n [pi_n * (demand[(n,k)] - Y_n)]
                              - sum_n [(mu + pi_n) * x_n]
        
        This is a valid lower bounding cut (Benders cut) for the recourse cost.
    """
    sp = gp.Model(f"Recourse_SP_{k}")
    sp.Params.OutputFlag = 0  # Suppress Gurobi solver console output

    # Decision variables for recourse problem
    y = {}           # Recourse supply sent to each city
    left = {}        # Leftover (excess) inventory at each city
    short = {}       # Shortage (unmet demand) at each city
    
    for n in cities:
        y[n] = sp.addVar(lb=0.0, name=f"y_{n}")              # recourse supply (y >= 0)
        left[n] = sp.addVar(lb=0.0, name=f"left_{n}")        # excess inventory (left >= 0)
        short[n] = sp.addVar(lb=0.0, name=f"short_{n}")      # unmet demand (short >= 0)
    sp.update()
    
    # Constraint C1: Capacity constraint for recourse supply
    # Interpretation: can only distribute what remains of budget after first-stage allocation
    cap_expr = gp.quicksum(y[n] for n in cities)
    remaining_budget = I - gp.quicksum(xvals[n] for n in cities)
    c1 = sp.addConstr(cap_expr <= remaining_budget, name="CapacityConstr")
    
    # Constraint C2: Demand balance for each city
    # Interpretation: inventory balance must hold (what you have = what's needed)
    city_constr = {}
    for n in cities:
        # Left side: what you have (starting + first-stage + second-stage)
        supply_available = Yn[n] + xvals[n] + y[n]
        # Right side: what's needed (demand + adjustment for excess/shortage)
        demand_and_adjustment = demand[(n, k)] + left[n] - short[n]
        # Set up equality: Y_n + x_n + y_n = demand_n + left_n - short_n
        city_constr[n] = sp.addConstr(supply_available == demand_and_adjustment, 
                                     name=f"CityConstr_{n}")
    
    # Objective function: minimize total recourse cost
    # Cost = supply cost + leftover penalty + shortage penalty
    obj_expr = gp.quicksum(theta_s[n]*y[n] + h*left[n] + g*short[n] for n in cities)
    sp.setObjective(obj_expr, GRB.MINIMIZE)
    sp.update()
    sp.optimize()
    
    # Extract solution values
    Q_val = sp.ObjVal  # Optimal recourse cost for this scenario and x
    
    # Extract dual variables (shadow prices) using .Pi attribute
    # These duals are used to generate Benders cuts
    mu = c1.Pi         # Dual variable for capacity constraint (must be >= 0)
    pi = {n: city_constr[n].Pi for n in cities}  # Dual variables for city constraints
    
    return Q_val, mu, pi

# --------------------------
# 3. Multiple-Cut Benders with Warm Start for Q1
# --------------------------
def benders_multicut_q1(max_iters=200, tol=tol):
    """
    Multiple-Cut Benders Decomposition Algorithm for Q1.
    
    Problem Being Solved:
    ======================
    Two-Stage Stochastic Program:
        Minimize: (sum over cities n of) [theta1[n] * x_n]
                + Expected_over_scenarios_k [(sum over cities n of) [theta_s[n]*y_n[k] + h*left_n[k] + g*short_n[k]]]
        
        Subject to:
            First-stage constraints:
                (sum over cities n of) x_n <= I  (distribute up to budget I)
                x_n >= 0 for all cities n
            
            Second-stage constraints (for each scenario k):
                (sum over cities n of) y_n[k] <= I - (sum over cities n of) x_n  (recourse capacity)
                For each city n:
                    Y_n + x_n + y_n[k] = demand[(n,k)] + left_n[k] - short_n[k]  (demand balance)
                y_n[k], left_n[k], short_n[k] >= 0
    
    Decision Variables in Master Problem:
    =====================================
    x[n]: First-stage allocation to city n (units to send from depot in stage 1)
          Decided "here-and-now" before demand is revealed
    
    eta[k]: Recourse cost approximation for scenario k
            Lower bound on Q_k(x) that gets tightened by Benders cuts
            Initially unbounded below, constrained by cuts as algorithm proceeds
    
    Master Problem Formulation:
    ===========================
    Minimize: (sum over cities n of) [theta1[n] * x_n]
            + (sum over scenarios k of) [prob * eta[k]]
    
    Subject to:
        Budget constraint:
            (sum over cities n of) x_n <= I
        
        Initial Benders cuts (from warm start):
            eta[k] >= Q_k(0) for each scenario k
        
        Additional Benders cuts (added during iterations):
            eta[k] >= mu * I + sum_n [pi_n * (demand[(n,k)] - Y_n)] - sum_n [(mu + pi_n) * x_n]
            for each scenario k and iteration
    
    Algorithm Flow:
    ===============
    
    1. INITIALIZATION
       - Create master problem with variables x[n] and eta[k]
       - Set initial constraints (budget, non-negativity)
       - Add master objective (first-stage + expected recourse)
    
    2. WARM START PHASE
       - Set x = 0 (no first-stage action)
       - For each scenario k:
         a) Solve recourse subproblem with x=0 to get Q_k(0) and duals (mu0, pi0)
         b) Add constraint: eta[k] >= Q_k(0) and set eta[k].LB = Q_k(0)
         c) This initializes eta variables with valid lower bounds
    
    3. MAIN BENDERS LOOP (iterate until convergence)
       For iteration iter = 1, 2, 3, ..., max_iters:
       
       a) SOLVE MASTER PROBLEM
          - Solve: minimize first-stage + expected recourse subject to current constraints
          - Extract: Lower Bound (LB) = master objective value
          - Extract: Current solution x* = {x[n].X for each city n}
          - Extract: Current recourse bounds eta* = {eta[k].X for each scenario k}
       
       b) EVALUATE RECOURSE AND GENERATE CUTS
          For each scenario k:
             i) Solve recourse subproblem at x* to get:
                - Q_k(x*) = true recourse cost at current x*
                - (mu*, pi*) = dual solution
             
             ii) Form Benders cut:
                 cut_rhs = mu* * I + sum_n [pi*_n * (demand[(n,k)] - Y_n)]
                 cut_expr = cut_rhs - sum_n [(mu* + pi*_n) * x_n]
             
             iii) Check if cut is violated:
                  current_cut_value = cut_rhs - sum_n [(mu* + pi*_n) * x*_n]
                  if eta*[k] < current_cut_value - tolerance:
                      add constraint: eta[k] >= cut_expr to master problem
                      set new_cut_added = True
       
       c) COMPUTE UPPER BOUND
          - Using current solution x*, evaluate all recourse costs: Q_k(x*) for each k
          - Compute: UB = sum_n [theta1[n] * x*_n] + sum_k [prob * Q_k(x*)]
          - This is a feasible solution value (valid upper bound on optimum)
          - Track: best_UB = minimum upper bound found so far
       
       d) CHECK CONVERGENCE
          - Compute relative optimality gap: gap = (best_UB - LB) / |LB|
          - If no new cuts added in this iteration:
              -> All eta[k] are tight (equal to recourse cuts)
              -> x* is optimal (convergence achieved)
              -> STOP
          - Else if gap < tolerance:
              -> Solution is sufficiently close to optimal
              -> STOP
          - Else:
              -> Continue to next iteration
    
    Intuition:
    ==========
    - LB increases (tightens) as we add more cuts (master gets harder)
    - UB decreases (improves) as we find better feasible solutions
    - When LB = UB (approximately), we have optimal solution
    - Warm start helps by providing initial valid cuts, speeding convergence
    
    Parameters:
    ===========
    max_iters: Maximum number of Benders iterations (safety limit to prevent infinite loops)
    tol: Relative optimality gap tolerance (stop if gap < tol)
    
    Returns:
    ========
    best_x_sol: Dictionary {city_n: x_n*} of optimal first-stage allocations
    best_UB: Best (smallest) upper bound found during algorithm
    LB: Final lower bound from master problem at termination
    """
    start_time = time.time()

    # ===== BUILD MASTER PROBLEM =====
    MP = gp.Model("Master_Multicut_Q1")
    MP.Params.OutputFlag = 0  # Suppress solver output

    # First-stage decision variables
    # x[n] = units to allocate to city n in stage 1 (from depot)
    x = {}
    for n in cities:
        x[n] = MP.addVar(lb=0.0, name=f"x_{n}")
    
    # First-stage budget constraint
    # Interpretation: total first-stage allocation cannot exceed available budget I
    MP.addConstr(gp.quicksum(x[n] for n in cities) <= I, name="Capacity")

    # Second-stage cost approximation variables
    # eta[k] is a lower bound on the recourse cost Q_k(x) for scenario k
    # It will be tightened by Benders cuts as the algorithm proceeds
    eta = {}
    for k in scenarios:
        eta[k] = MP.addVar(lb=0.0, name=f"eta_{k}")
    
    # Master objective function
    # Minimize: first-stage cost + expected second-stage cost
    # Expected second-stage cost = sum_k [prob(k) * eta[k]]
    first_stage_obj = gp.quicksum(theta1[n]*x[n] for n in cities)
    second_stage_obj = gp.quicksum(prob*eta[k] for k in scenarios)
    MP.setObjective(first_stage_obj + second_stage_obj, GRB.MINIMIZE)
    MP.update()

    # ===== WARM START PHASE =====
    # Initialize with x=0 and compute Q_k(0) for each scenario
    print("Warm Start Phase: Computing Q_k(0) and initializing eta bounds...")
    x0 = {n: 0.0 for n in cities}  # Start with no first-stage allocation
    
    for k in scenarios:
        # Solve recourse problem with x=0
        Q0, mu0, pi0 = solve_recourse_subproblem(k, x0)
        
        # Set lower bound on eta[k] to be Q0
        # This ensures eta[k] >= Q_k(0) for all iterations
        if Q0 > tol:
            eta[k].LB = Q0
            # Also explicitly add constraint for clarity
            MP.addConstr(eta[k] >= Q0, name=f"WarmStartCut_{k}")
    
    MP.update()

    # ===== MAIN BENDERS LOOP =====
    best_UB = float('inf')  # Track best (smallest) upper bound found
    best_x_sol = None       # Track best first-stage solution
    cut_count = 0           # Count total Benders cuts added
    iter_count = 0          # Count iterations

    while iter_count < max_iters:
        iter_count += 1
        
        # ===== STEP 3a: SOLVE MASTER PROBLEM =====
        MP.optimize()
        
        if MP.Status != GRB.OPTIMAL:
            print("Master problem not optimal. Stopping.")
            break
        
        # Extract lower bound and current solution from master
        LB = MP.ObjVal  # Lower Bound from master problem
        xvals = {n: x[n].X for n in cities}  # Current first-stage solution
        eta_vals = {k: eta[k].X for k in scenarios}  # Current eta values

        # ===== STEP 3b: EVALUATE RECOURSE AND GENERATE CUTS =====
        recourse_total = 0.0  # Accumulate expected recourse cost for upper bound
        new_cut_added = False  # Track if any cut was added this iteration
        
        for k in scenarios:
            # Solve recourse subproblem at current solution xvals
            Qk, mu, pi = solve_recourse_subproblem(k, xvals)
            recourse_total += prob * Qk  # Add weighted recourse cost
            
            # Construct Benders cut using dual solution
            # Cut RHS: mu * I + sum_n [pi_n * (demand_nk - Y_n)]
            cut_const = mu * I + sum(pi[n]*(demand[(n, k)] - Yn[n]) for n in cities)
            
            # Evaluate cut at current solution xvals
            # Cut evaluates to: cut_const - sum_n [(mu + pi_n) * x_n]
            current_cut_value = cut_const - sum((mu + pi[n]) * xvals[n] for n in cities)
            
            # Check if cut is violated at current solution
            # Cut violated means: eta[k] < current_cut_value - tolerance
            if eta_vals[k] < current_cut_value - tol:
                # Add new Benders cut to master problem
                # Constraint: eta[k] >= cut_const - sum_n [(mu + pi_n) * x_n]
                expr = gp.LinExpr(cut_const)
                for n in cities:
                    expr -= (mu + pi[n]) * x[n]
                MP.addConstr(eta[k] >= expr, name=f"BendersCut_{k}_iter{iter_count}")
                new_cut_added = True
                cut_count += 1
        
        # ===== STEP 3c: COMPUTE UPPER BOUND =====
        # Upper bound = feasible solution value = first-stage + expected recourse
        first_stage_cost = sum(theta1[n]*xvals[n] for n in cities)
        UB = first_stage_cost + recourse_total
        
        # Update best upper bound if current is better
        if UB < best_UB:
            best_UB = UB
            best_x_sol = xvals.copy()
        
        # ===== STEP 3d: CHECK CONVERGENCE =====
        gap = abs(best_UB - LB) / (abs(LB) + 1e-9)  # Relative optimality gap
        
        print(f"Iteration {iter_count}: "
              f"LB = {LB:.4f}, UB = {best_UB:.4f}, "
              f"Gap = {gap*100:.2f}%, "
              f"Total Cuts = {cut_count}, "
              f"New Cuts = {1 if new_cut_added else 0}")
        
        # Termination condition 1: No new cuts added
        if not new_cut_added and iter_count > 1:
            print("-> No new cuts added. Convergence achieved (eta variables are tight).")
            break
        
        # Termination condition 2: Gap below tolerance
        if gap < tol:
            print(f"-> Relative gap {gap*100:.2f}% is below tolerance {tol*100:.2f}%. Stopping.")
            break
        
        MP.update()

    # ===== PRINT FINAL RESULTS =====
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("MULTI-CUT BENDERS DECOMPOSITION FOR Q1 - FINAL RESULTS")
    print("="*70)
    print(f"Total Iterations Completed: {iter_count}")
    print(f"Total Benders Cuts Added: {cut_count}")
    print(f"Elapsed Computation Time: {elapsed:.2f} seconds")
    print()
    print(f"Final Lower Bound (LB): {LB:.6f}")
    print(f"Final Upper Bound (UB): {best_UB:.6f}")
    print(f"Final Optimality Gap: {gap*100:.4f}%")
    print()
    print(f"Optimal First-Stage Solution x*:")
    for n in cities:
        if best_x_sol[n] > 1e-6:
            print(f"  x[{n}] = {best_x_sol[n]:.4f} units")
    print("="*70 + "\n")
    
    return best_x_sol, best_UB, LB

# --------------------------
# Main Section
# --------------------------
if __name__ == "__main__":
    print("="*70)
    print("RUNNING MULTI-CUT BENDERS DECOMPOSITION FOR ASSIGNMENT 3, Q1")
    print("="*70 + "\n")
    
    best_x_sol, best_UB, final_LB = benders_multicut_q1(max_iters=200, tol=tol)
    
    print("\nFINAL SUMMARY")
    print("-" * 70)
    print(f"Optimal First-Stage Solution: {best_x_sol}")
    print(f"Final Lower Bound: {final_LB:.6f}")
    print(f"Final Upper Bound: {best_UB:.6f}")
    print(f"Optimality Gap: {abs(best_UB - final_LB)/abs(final_LB)*100:.4f}%")
    print("-" * 70)