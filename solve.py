# Module contains a set of subroutines for solving systems of equations numerically.

import math, pickle, warnings, time, json, copy
import multiprocessing as mp
import sympy as sy
import numpy as np
import scipy.optimize as so
from solution import Solution

_lambdified = []

def _evaluate_wrapper(values, variables, frozen, lambdified):
    """
    Wrapper over evaluate() function. Returns list of evaluated values
    for lambdified equations.

    Parameters
    ----------
    values:     list of initial values for independent variables in equations
    variables:  list of independent variables (of type sympy.Symbol)
    frozen:     dict of form variable: frozen value
    lambdified: list of pairs (lambdified equation, variables)
    """
    values = map(float, values)
    value_dict = dict({s:v for s, v in zip(variables, values)}, **frozen)
    # print "value_dict:", value_dict
    return evaluate(value_dict, lambdified=lambdified)

def _evaluate_single(val, var, lambdified, frozen):
    """
    Evaluates single equation numerically if it contains only one variable.

    Parameters
    ----------
    val:        initial value of variable var
    lambdified: pair (tuple or list) of form: (lambdified equation, variables)
    var:        single variable in equation (of type sympy.Symbol)
    frozen:     dict of form (variable: frozen value)
    """
    func, param_str = lambdified
    value_dict = dict({var: val}, **frozen)
    args = [value_dict[solvar] for solvar in param_str]
    try:
        res = func(*args)
    except:
        res = 1e10
    return res

def _iterative_solve(sol_inits, sol_vars, frozen, solve_order):
    """
    Solves system of equations itteratively, assuming it contains 
    single-variable equations.

    Parameters
    ----------
    sol_inits: list of initial values for independent variables in equations
    sol_vars:    corresponding list of independent variables
    frozen:      dict of form (variable: frozen value)
    solve_order: dict of form 
                 (number of variables in equation: (index of equation, variables))

    """
    resolved = {}
    iter_num = len(_lambdified)
    order = 1
    # solved_vars = []
    rsid = []
    for n in range(iter_num):
        suspects = solve_order.get(order, [])
        # print "suspects:", suspects
        for case in suspects:
            i, variables = case
            # unsolved = list(set(variables) - set(solved_vars))
            unsolved = list(variables - set(resolved.keys()))
            if len(unsolved) == 1:
                var = unsolved[0]
                ii = sol_vars.index(var)
                inval, lamd = sol_inits[ii], _lambdified[i]
                res_froz = dict(frozen, **resolved)
                # res = so.fsolve(_evaluate_single, inval, args=(lamd, var, res_froz))
                # resolved[var] = res[0]
                res = _fsolve(_evaluate_single, inval, True, var, lamd, res_froz)
                resolved[var] = res[var]

                rsid.append(i)
                # solved_vars.append(solvar)
        order += 1
    return resolved, rsid

def _fsolve(eval_func, init_values, single, *args):
    if single: 
        sol_vars = [args[0]]
    else: 
        sol_vars = args[0]

    res_ = so.fsolve(eval_func, init_values, args=args, full_output=True)

    if res_[-2] == 1:
        res = res_[0]
        res_dict = {v: res[i] for i, v in enumerate(sol_vars)}
    else:
        res = []
        res_dict = {v: None for i, v in enumerate(sol_vars)}
    return res_dict

def _find_solutions(input_list):
    """
    Solves system of equations in a range of given initial values of independent 
    variables for a fixed values of frozen variables.

    Parameters
    ----------
    input_list: list that contains next parameters in strict order:
        
        sweep_values: list of values of frozen variables
        sweep_vars:   list of frozen variables
        sol_vars:     list of independent variables
        sol_inits:    list of intitial values of independent variables

    """

    sweep_values, sweep_vars, sol_inits, sol_vars = input_list
    reslist = []
    frozen = {ss: sv for ss, sv in zip(sweep_vars, sweep_values)}
    # tol_res = []
    for vec in sol_inits:
        sol_res = fsolve(vec, sol_vars, frozen)
        reslist.append(Solution(frozen, sol_res))
    return reslist

def _sweep_singlecore(sweep_array, sweep_vars, sol_inits, sol_vars):
    results = []
    for sweep_values in sweep_array:
        inputlist = [sweep_values, sweep_vars, sol_inits, sol_vars]
        res = _find_solutions(inputlist)
        results.append(res)
    return results

def _sweep_multicore(sweep_array, sweep_vars, sol_inits, sol_vars):
    inputlist = [sweep_vars, sol_inits, sol_vars]
    corenums = mp.cpu_count()
    if corenums > 1: corenums -= 1

    pool = mp.Pool(processes=corenums)
    joblist = [[sweep_values] + inputlist for sweep_values in sweep_array]
    mapresult = pool.map_async(_find_solutions, joblist)
    pool.close()
    pool.join()

    results = []
    for reslist in mapresult.get():
        results.append(reslist)
    return results

def sweep_solve(sweep_array, sweep_vars, sol_inits, sol_vars, multithread):
    """
    Solves system of equations in a range of frozen variables.

    Parameters
    ----------

    sweep_array:  list of lists of frozen variables values
    sweep_vars:   list of frozen variables
    sol_inits:    list of intitial values of independent variables
    sol_vars:     list of independent variables
    multithread:  bool, whether or not use multithreading

    """

    sweepfunc = _sweep_multicore if multithread else _sweep_singlecore
    return sweepfunc(sweep_array, sweep_vars, sol_inits, sol_vars)

def evaluate(value_dict, lambdified=None):
    """
    Evaluates equations for a given values of variables.

    Parameters
    ----------
    value_dict: dict of form (variable: value)
    lambdified: pair (tuple or list) of form: (lambdified equation, variables)

    """

    if lambdified is None:
        lambdified = _lambdified
    result = []
    for func, param_str in lambdified:
        args = [value_dict[varstr] for varstr in param_str]
        try:
            res = func(*args)
        except:
            res = 1e10
        result.append(res)
    
    if len(value_dict) == 1:
        return result[0]
    else:
        return result

def fsolve(sol_inits, sol_vars, frozen, solve_order=None):
    """
    Solves system of equations numerrically.

    Parameters
    ----------
    sol_inits:   list of intitial values of independent variables
    sol_vars:    list of independent variables
    frozen:      dict of form (variable: frozen value)
    solve_order: dict of form 
                 (number of variables in equation: (index of equation, variables))

    """
    
    lambdified = _lambdified

    if solve_order is not None:
        if 1 in solve_order:
            resolved, rsid = _iterative_solve(sol_inits, sol_vars, frozen, solve_order)
            if len(resolved) != len(sol_vars):
                lambdified = []
                for i, lamd in enumerate(_lambdified):
                    if i not in rsid: lambdified.append(lamd)
                frozen = dict(frozen, **resolved)
            else:
                return resolved

    res_dict = _fsolve(_evaluate_wrapper, sol_inits, False, sol_vars, frozen, lambdified)
    return res_dict



