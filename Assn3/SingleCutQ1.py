# -*- coding: utf-8 -*-
###############################################################################
# Benders_SingleCut_Q1_Refined_Corrected.py
#
# Single-cut Benders Decomposition for Assignment 3, Q1, with corrected cut.
# Derivation assumes the subproblem formulation provided in this file.
###############################################################################

import gurobipy as gp
from gurobipy import GRB
import time # Added for timing

# --- Data reading (Assume ReadData.py exists and works) ---
# Ensure ReadData.py is in the same directory or Python path
try:
    import ReadData
    cities = ReadData.cities         # e.g., ['C0','C1',...,'C9']
    scenarios = ReadData.scenarios   # e.g., ['S0', 'S1', ...]
    prob = 1.0 / len(scenarios)      # assume equally likely
    I = ReadData.I                   # e.g., 500.0
    Yn = ReadData.Yn                 # initial inventory dict {city: value}
    demand = ReadData.demand         # demand dict {(city, scenario): value}
    theta_1 = ReadData.theta         # cost for first-stage x dict {city: value}
    theta_2 = ReadData.theta_s       # cost for second-stage (u+v) dict {city: value}
    h = ReadData.h                   # leftover cost (scalar)
    g = ReadData.g                   # shortage cost (scalar)
    print("Data loaded successfully from ReadData.py")
except ImportError:
    print("Error: ReadData.py not found. Please ensure it's in the correct directory.")
    print("Using placeholder data for demonstration - RESULTS WILL BE MEANINGLESS.")
    # Example Placeholder Data (Adapt if your structure differs)
    cities = [f'C{i}' for i in range(5)] # Reduced size for example
    scenarios = [f'S{k}' for k in range(3)]
    prob = 1.0 / len(scenarios)
    I = 200.0
    Yn = {n: 30.0 for n in cities}
    demand = {(n, k): 40.0 + i*1.0 + j*2.0 for i,n in enumerate(cities) for j,k in enumerate(scenarios)}
    theta_1 = {n: 2.0 for n in cities}
    theta_2 = {n: 3.0 for n in cities} # Assuming theta_s is the cost for u+v
    h = 0.5
    g = 5.0
except Exception as e:
    print(f"Error loading data from ReadData.py: {e}")
    # Exit if data loading fails fundamentally
    import sys
    sys.exit(1)


# Tolerance for cut violation and zero checks
CutViolationTolerance = 1e-6

#-----------------------------------------------------------------------------
# Subproblem builder: "BuildAndSolveSubproblem"
# (Using the formulation provided in the original script)
#-----------------------------------------------------------------------------
def BuildAndSolveSubproblem(k, xvals):
    """
    Solves the second-stage subproblem for scenario k, given xvals.
    Uses the variable definitions (u, v, z_, s_) and constraints
    as defined in the original script.

    Returns:
        Qk_val: Optimal objective value for scenario k.
        pi_c1: Dual variable for the "big_constraint" (c1.Pi, >= 0).
        pi_c2: Dictionary of dual variables for demand constraints {n: c2[n].Pi}.
    """
    # --- Input Data Validation ---
    if not isinstance(xvals, dict):
        print(f"Error in SP {k}: xvals is not a dictionary.")
        return (float('inf'), 0, {n: 0 for n in cities})
    current_demand = {(n): demand.get((n, k), None) for n in cities}
    if any(d is None for d in current_demand.values()):
         missing_cities = [n for n, d in current_demand.items() if d is None]
         print(f"Error in SP {k}: Missing demand data for cities {missing_cities} in scenario {k}.")
         return (float('inf'), 0, {n: 0 for n in cities})
    # --- End Validation ---

    sp = gp.Model(f"Subproblem_{k}")
    sp.Params.OutputFlag = 0  # Mute solver logs unless debugging

    # second-stage variables
    u = {}
    v = {}
    z_ = {}
    s_ = {}
    for n in cities:
        # Use get() for robustness if Yn doesn't contain all cities (should not happen with good data)
        if Yn.get(n) is None: print(f"Warning: Missing Yn for city {n} in SP {k}")
        u[n]  = sp.addVar(lb=0.0, name=f"u_{n}_{k}")
        v[n]  = sp.addVar(lb=0.0, name=f"v_{n}_{k}")
        z_[n] = sp.addVar(lb=0.0, name=f"z_{n}_{k}")
        s_[n] = sp.addVar(lb=0.0, name=f"s_{n}_{k}")
    sp.update() # Needed before using variables in constraints

    # "Big" constraint: sum(u) - sum(v) - sum(xvals) + I >= 0
    sum_x_val = sum(xvals.get(n, 0.0) for n in cities) # Use get() for safety
    lhs_uv = gp.quicksum(u[n] for n in cities) \
             - gp.quicksum(v[n] for n in cities) \
             - sum_x_val
    try:
        # Added check for I being a number
        if not isinstance(I, (int, float)): raise TypeError("I is not a number")
        c1 = sp.addConstr(lhs_uv + I >= 0.0, name="big_constraint")
    except (TypeError, gp.GurobiError) as e:
        print(f"Error creating constraint c1 in SP {k}: {e}")
        return (float('inf'), 0, {n: 0 for n in cities})

    # City constraints: Yn[n] + xvals[n] + v[n] + s_[n] - z_[n] - u[n] = demand
    c2 = {}
    try:
        for n in cities:
            # Use get() for Yn and xvals
            lhs = Yn.get(n, 0.0) + xvals.get(n, 0.0) + v[n] + s_[n] - z_[n] - u[n]
            # Demand for (n, k) was checked previously
            c2[n] = sp.addConstr(lhs == demand[(n, k)], name=f"demand_{n}_{k}")
    except (TypeError, gp.GurobiError) as e:
         print(f"Error creating constraint c2 for city {n} in SP {k}: {e}")
         return (float('inf'), 0, {n: 0 for n in cities})


    # Objective
    try:
        obj_expr = gp.quicksum(theta_2.get(n, 0.0)*(u[n]+v[n]) + h*z_[n] + g*s_[n]
                                for n in cities)
        sp.setObjective(obj_expr, GRB.MINIMIZE)
    except (TypeError, gp.GurobiError) as e:
        print(f"Error setting objective in SP {k}: {e}")
        return (float('inf'), 0, {n: 0 for n in cities})

    # Solve
    try:
        sp.optimize()
    except gp.GurobiError as e:
        print(f"Gurobi error during optimization of SP {k}: {e}")
        return (float('inf'), 0, {n: 0 for n in cities})


    # --- Process Results ---
    if sp.Status == GRB.OPTIMAL:
        Qk_val = sp.ObjVal
        # Check if duals exist (can be missing if presolve found optimality)
        try:
            pi_c1 = c1.Pi
            pi_c2 = {n: c2[n].Pi for n in cities}
        except AttributeError: # Handle cases where .Pi might not be available
             print(f"Warning: Could not retrieve duals for SP {k} (Status Optimal). Using zeros.")
             pi_c1 = 0.0
             pi_c2 = {n: 0.0 for n in cities}
             # Or try resolving with dual reductions off: sp.Params.DualReductions = 0; sp.optimize()
        return (Qk_val, pi_c1, pi_c2)
    elif sp.Status == GRB.INFEASIBLE:
        print(f"Subproblem {k} is INFEASIBLE for x={ {n:v for n,v in xvals.items() if abs(v)>1e-9} }") # Print non-zero x
        # sp.computeIIS() # Compute Irreducible Inconsistent Subsystem
        # sp.write(f"subproblem_{k}_iis.ilp")
        # print(f"IIS written to subproblem_{k}_iis.ilp")
        return (float('inf'), 0, {n: 0 for n in cities}) # Indicate infeasibility
    else:
        print(f"Subproblem {k} did not solve optimally (Status code: {sp.Status})")
        return (float('inf'), 0, {n: 0 for n in cities}) # Treat as infeasible


#-----------------------------------------------------------------------------
# Single-Cut Benders with corrected cut calculation
#-----------------------------------------------------------------------------
def SingleCutBendersQ1_RefinedWarmCorrected():
    """
    Single-cut Benders with corrected cut calculation.
    Cut: theta >= sum_k p_k * [ (pi_c1*I + sum_n pi_c2[n]*(d_nk - Y_n))
                              + sum_n (pi_c2[n] - pi_c1)*x[n] ]
    """
    print("Starting Single-Cut Benders (Corrected Cut)...")
    start_time = time.time()

    # --- Master Problem Setup ---
    try:
        MP = gp.Model("Master_SingleCut_Q1")
        # MP.Params.OutputFlag = 0  # Mute logs unless debugging
        MP.Params.Method = 1      # Use dual simplex, often good for Benders

        # x variables
        x = {}
        for n in cities:
            x[n] = MP.addVar(lb=0.0, name=f"x_{n}")
        # Capacity constraint: sum_n x[n] <= I
        MP.addConstr(gp.quicksum(x[n] for n in cities) <= I, name="Capacity")

        # single variable for expected recourse cost: theta
        theta = MP.addVar(lb=0.0, name="theta") # Theta >= 0 initially

        # objective = sum_n theta_1[n]* x[n] + theta
        obj_expr = gp.quicksum(theta_1.get(n, 0.0)*x[n] for n in cities) + theta # Use get()
        MP.setObjective(obj_expr, GRB.MINIMIZE)
        MP.update()
    except (TypeError, gp.GurobiError, Exception) as e:
        print(f"Fatal Error: Failed to build master problem: {e}")
        return None, float('inf'), -float('inf')
    # --- End Master Setup ---


    # --- Warm Start ---
    print("Running Warm Start (x=0)...")
    x0 = {n:0.0 for n in cities}
    sum_Q0 = 0.0
    agg_const_warm = 0.0
    agg_alpha_warm = {n: 0.0 for n in cities}
    warm_start_feasible = True

    for k in scenarios:
        Qk_val, pi_c1, pi_c2 = BuildAndSolveSubproblem(k, x0)

        if Qk_val == float('inf'):
             print(f"ERROR: Subproblem {k} infeasible even for x=0. Check data/model.")
             warm_start_feasible = False
             break # Cannot generate valid warm start cut

        sum_Q0 += prob * Qk_val

        # Calculate terms for CORRECTED warm start cut
        const_k_corr = pi_c1 * I
        for nn in cities:
            const_k_corr += pi_c2.get(nn, 0.0) * (demand.get((nn, k), 0.0) - Yn.get(nn, 0.0)) # Use get()
        agg_const_warm += prob * const_k_corr

        for nn in cities:
            slope_k_nn_corr = pi_c2.get(nn, 0.0) - pi_c1 # Use get()
            agg_alpha_warm[nn] += prob * slope_k_nn_corr

    if not warm_start_feasible:
         print("Cannot proceed due to warm start infeasibility.")
         return None, float('inf'), -float('inf')

    print(f"Warm Start E[Q(0)] = {sum_Q0:.6f}")

    # If sum_Q0 > tol, add the warm start cut
    if sum_Q0 > CutViolationTolerance:
        print("Adding Warm Start Cut...")
        try:
            RHS_expr_warm = gp.LinExpr(agg_const_warm) # Start with constant
            for nn in cities:
                RHS_expr_warm += agg_alpha_warm.get(nn, 0.0) * x[nn] # Use get()
            MP.addConstr(theta >= RHS_expr_warm, name="WarmStartCut")
            MP.update()
        except (TypeError, gp.GurobiError) as e:
             print(f"Error adding warm start cut: {e}")
             return None, float('inf'), -float('inf') # Cannot proceed if cut fails
    else:
         print("Warm Start E[Q(0)] is near zero, no initial cut needed.")
    # --- End Warm Start ---


    # --- Benders Loop ---
    CutFound = True
    NoIters = 0
    BestUB = float('inf')
    LB = -float('inf')
    best_x_sol_for_UB = None # Store the x solution that yielded the best UB

    while CutFound and NoIters < 200: # Iteration limit
        NoIters += 1
        CutFound = False # Assume no cut needed until violation found
        print(f"\n--- Iteration {NoIters} ---")

        # --- Solve Master Problem ---
        try:
            MP.optimize()
        except gp.GurobiError as e:
            print(f"Gurobi error optimizing master problem: {e}")
            break # Stop loop if master solve fails

        if MP.Status == GRB.OPTIMAL or MP.Status == GRB.SUBOPTIMAL:
            LB = MP.ObjVal # Update Lower Bound
            # Ensure solution values are retrieved safely
            try:
                 xsol = {n: x[n].X for n in cities}
                 thetasol = theta.X
            except AttributeError:
                 print("Error retrieving solution values from master problem. Status was optimal/suboptimal but X missing?")
                 break
            print(f"Master solved. Current LB = {LB:.6f}, theta = {thetasol:.6f}")
            # Optional sanity check
            current_sum_x = sum(xsol.values())
            if current_sum_x > I + CutViolationTolerance:
                 print(f"WARNING: Master solution violates capacity! Sum x = {current_sum_x:.6f} > I = {I}")
                 # Depending on the problem, you might stop or just note this
        elif MP.Status == GRB.INFEASIBLE:
             print("Master problem INFEASIBLE. Stopping. Previous LB was:", LB)
             print("Check added cuts. Computing IIS...")
             MP.computeIIS()
             MP.write("master_iis.ilp")
             print("IIS written to master_iis.ilp")
             LB = float('inf') # Infeasible master means LB is infinite (for minimization)
             break
        elif MP.Status == GRB.UNBOUNDED:
             print("Master problem UNBOUNDED. Stopping. Likely missing constraints or cuts.")
             # This typically shouldn't happen if theta >= 0 and cuts are added correctly.
             LB = -float('inf') # Unbounded master -> LB is -infinity
             break
        else:
            print(f"Master problem FAILED to solve optimally (Status: {MP.Status}). Stopping.")
            break # Stop if master fails for other reasons
        # --- End Solve Master ---

        # --- Solve Subproblems & Calculate UB ---
        FirstStageCost = sum(theta_1.get(n_, 0.0)*xsol.get(n_, 0.0) for n_ in cities) # Use get()
        ExpectedRecourseCost_Actual = 0.0
        agg_const_cut = 0.0
        agg_alpha_cut = {n: 0.0 for n in cities}
        subproblem_infeasible_in_loop = False

        #print("Solving subproblems...")
        for k in scenarios:
            Qk_val, pi_c1, pi_c2 = BuildAndSolveSubproblem(k, xsol)

            if Qk_val == float('inf'):
                 print(f"ERROR: Subproblem {k} infeasible for xsol. Benders cannot proceed without feasibility cuts.")
                 subproblem_infeasible_in_loop = True
                 break

            ExpectedRecourseCost_Actual += prob * Qk_val

            # Accumulate terms for the CORRECTED cut
            const_k_corr = pi_c1 * I
            for nn in cities:
                 const_k_corr += pi_c2.get(nn, 0.0) * (demand.get((nn, k), 0.0) - Yn.get(nn, 0.0))
            agg_const_cut += prob * const_k_corr

            for nn in cities:
                 slope_k_nn_corr = pi_c2.get(nn, 0.0) - pi_c1
                 agg_alpha_cut[nn] += prob * slope_k_nn_corr

        if subproblem_infeasible_in_loop:
             print("Stopping loop due to subproblem infeasibility.")
             # Keep current LB, but UB cannot be calculated or improved this iteration
             # BestUB remains the best valid UB found so far.
             # Depending on the cause, may need feasibility cuts or model review.
             break # Exit Benders loop

        # Calculate current UB
        current_UB = FirstStageCost + ExpectedRecourseCost_Actual
        #print(f"Subproblems solved. Current UB = {current_UB:.6f} (FirstStage={FirstStageCost:.6f}, ExpRecourse={ExpectedRecourseCost_Actual:.6f})")

        # Update best UB found so far and store the corresponding x
        if current_UB < BestUB:
            BestUB = current_UB
            best_x_sol_for_UB = xsol.copy() # Store the x that gave the best UB
            print(f"  *** New Best UB found: {BestUB:.6f} (at Iter {NoIters}) ***")
        # --- End UB Calculation ---

        # --- Check Cut Violation & Add Cut ---
        if thetasol < ExpectedRecourseCost_Actual - CutViolationTolerance:
            #print(f"Violation detected: theta ({thetasol:.6f}) < E[Q(x)] ({ExpectedRecourseCost_Actual:.6f})")
            print("Adding Single Optimality Cut...")
            try:
                cut_expr_corr = gp.LinExpr(agg_const_cut)
                for nn in cities:
                    cut_expr_corr += agg_alpha_cut.get(nn, 0.0) * x[nn]

                MP.addConstr(theta >= cut_expr_corr, name=f"SingleCut_{NoIters}")
                CutFound = True
                MP.update()
            except (TypeError, gp.GurobiError) as e:
                print(f"Error adding cut in iteration {NoIters}: {e}")
                break # Stop if adding cut fails
        else:
             #print("No significant cut violation detected.")
             pass # No cut needed this iteration

        # --- Check Convergence ---
        gap = float('inf')
        # Use BestUB (best feasible solution found) and LB (current master relaxation)
        if LB > -float('inf') and BestUB < float('inf'):
            if abs(LB) > 1e-9: # Avoid division by zero if LB is near zero
                 gap = abs(BestUB - LB) / abs(LB)
            elif abs(BestUB) < 1e-9: # If both LB and UB are near zero
                 gap = 0.0
            # else: gap remains infinite (LB near zero, UB large)

        print(f"Iter {NoIters}: LB={LB:.6f}, Best UB={BestUB:.6f}, Gap={gap*100:.4f}%")

        if not CutFound and NoIters > 0: # Need NoIters > 0 as warm start might not need cut
             print("\nConvergence achieved: No new violated cut found.")
             break
        # Use a tolerance relative to UB as well, or absolute tolerance?
        # Using relative gap based on LB:
        if gap < CutViolationTolerance :
             print(f"\nConvergence achieved: Gap ({gap*100:.6f}%) is below tolerance ({CutViolationTolerance*100:.6f}%).")
             break
        if NoIters >= 200: # Max iteration check
             print("\nStopping: Maximum iterations reached.")
             break
    # --- End Benders Loop ---

    # --- Final Results Reporting ---
    elapsed = time.time() - start_time
    print('\n========================================')
    print('Benders Single-Cut Q1 final results:')
    print(f'Elapsed time: {elapsed:.2f} sec')
    print(f'Total Iterations: {NoIters}')

    # Final LB is the LB from the last successful master solve
    final_LB = LB
    # Final UB is the best feasible UB found across all iterations
    final_UB = BestUB

    # Handle cases where LB/UB might still be infinite if loop exited early/problematically
    if final_LB == -float('inf') or final_UB == float('inf'):
         print("Warning: Algorithm did not converge to finite bounds.")
         final_gap = float('inf')
    elif abs(final_LB) > 1e-9:
         final_gap = abs(final_UB - final_LB) / abs(final_LB)
    elif abs(final_UB) < 1e-9:
         final_gap = 0.0 # Both near zero
    else:
         final_gap = float('inf') # LB near zero, UB is finite -> infinite gap

    print(f'Final LB: {final_LB:.6f}')
    print(f'Best UB: {final_UB:.6f}')
    print(f"Final Gap: {final_gap * 100:.4f}%")

    # Print the solution associated with the Best UB found
    if best_x_sol_for_UB is not None:
        print('\nBest Feasible Solution Found (corresponds to Best UB):')
        print('xsol:')
        total_x_best = 0
        sorted_cities_best = sorted(best_x_sol_for_UB.keys())
        for n in sorted_cities_best:
             value = best_x_sol_for_UB.get(n, 0.0)
             if abs(value) > CutViolationTolerance: # Check absolute value for printing
                 print(f"  x['{n}'] = {value:.6f}")
                 total_x_best += value
        print(f"  Sum x = {total_x_best:.6f}")
    else:
        print("\nNo feasible solution found or stored (Best UB remained infinite).")

    print('========================================')

    # Return the solution corresponding to BestUB, BestUB, and FinalLB
    return best_x_sol_for_UB, final_UB, final_LB


# --- Main Execution Block ---
if __name__=="__main__":
    # Run the corrected single-cut approach
    finalX_best, finalUB, finalLB = SingleCutBendersQ1_RefinedWarmCorrected()

    print("\n--- Final Summary ---")
    print(f"Final LB = {finalLB:.6f}")
    print(f"Final UB = {finalUB:.6f}")

    # Print the final solution (associated with the best UB) again
    if finalX_best:
         print("\nBest x solution found (corresponding to Best UB):")
         total_x = 0
         # Sort cities for consistent output
         sorted_cities = sorted(finalX_best.keys())
         for n in sorted_cities:
             value = finalX_best.get(n, 0.0)
             if abs(value) > CutViolationTolerance: # Check absolute value for printing
                 print(f"  x['{n}'] = {value:.6f}") # Print x value for city 'n'
                 total_x += value
         print(f"Total units moved (Sum x): {total_x:.6f}")
    else:
         print("\nNo feasible x solution determined.")

    print("\nScript finished.")
# --- End of Script ---