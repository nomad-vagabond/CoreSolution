import sys, os
import math, pickle, warnings, time, json, copy
import multiprocessing as mp
import sympy as sy
import numpy as np
import scipy.optimize as so
from solution import Solution

_error_msg = {
    0: 'init_vec parameter must represent initial values of variables',
    1: 'unknown variable found in the selection criterion expression: '
}

_info_msg = {
    0: "Number of solutions found: %d",
    1: "Sweep solve finished in %f seconds."
}


class Solver(object):
    """
    Numeric solver. Solves determined and handles parametric sweep for 
    undetermined systems of equations.

    Parameters
    ----------
    equations: A list of equations representing system of equations. All equations
               must be a SymPy objects of type Add, Mul or Pow.

    """
    def __init__(self, equations):

        self.equations = equations
        self.variables = []
        self.lambdified = []
        # self.solve_variables = {}
        self.solve_order = {}
        for i, eq in self.equations.items():
            variables = eq.free_symbols
            # self.solve_variables[i] = variables
            solorder = len(variables)
            if solorder not in self.solve_order:
                self.solve_order[solorder] = []
            self.solve_order[solorder].append((i, variables))
            self.lambdified.append([sy.lambdify(variables, eq), variables])
            self.variables += variables
        self.variables = set(self.variables)
        self.selector_map = {min: np.argmin, 'min': np.argmin, 
                             max: np.argmax, 'max': np.argmax}

    def _filter_solutions(self, reslist, constraints, comp_eps):
        filtered = []
        hasnone = False
        tol_res = []
        for res in reslist:
            if None in res.values():
                if not hasnone:
                    filtered.append(res)
                    hasnone = True
            else:
                res_ = Solution({var: float(val) for var, val in res.items()})
                # res_ = res
                res_int = [int(round(r/comp_eps)) for r in res_.values()]
                if res_int not in tol_res:
                    tol_res.append(res_int)

                    if len(constraints) > 0:
                        satisfy = True
                        for constr in constraints:
                            # c = const.subs(res_dict)
                            c = constr.subs(res_)
                            satisfy = satisfy & c
                        if satisfy:
                            filtered.append(res_)
                    else:
                        filtered.append(res_)
        return filtered

    def _select_solution(self, reslist, expression, criterion, system):
        if system is not None:
            redundant_varset = set(system.redundant.keys())

        target = []
        hasnone = False
        nonempty = []

        for res in reslist:
            if None in res.values(): 
                hasnone = True
            else:
                res_ = res
                variables = expression.free_symbols
                notinres = variables - set(res.keys())

                if len(notinres) > 0: 
                    if system is not None:
                        unknowns = len(notinres - redundant_varset)
                        if not unknowns:
                            res_ = system.adopt(res)[0]
                        else:
                            raise ValueError(_error_msg[1] + str(inredundant))
                    else:
                        raise ValueError(_error_msg[1] + str(notinres))

                subs_exp = expression.subs(res_).evalf()
                target.append(float(subs_exp))
                nonempty.append(res)

        if len(target) > 0:
            i = criterion(target)
            return [nonempty[i]]
        elif hasnone:
            return [reslist[0]]

    def _mix_inits(self, solve_inits, swnum):
        paramnum = len(self.variables) - swnum

        if solve_inits is None:
            par_mix = [[0.0]*paramnum]
            sol_vars = self.variables
        elif len(solve_inits) == paramnum:
            vs = [(k, v) for k, v in solve_inits.items()]
            sol_vars, init_values = zip(*vs)
            par_mix = np.array(np.meshgrid(*init_values)).T.reshape(-1, paramnum)
        else:
            raise ValueError(_error_msg[0])

        return par_mix, sol_vars

    def show_res(self, with_empty=False):
        if hasattr(self, "results"):
            rescount = 0
            for res_dict in self.results:
                # print "res type:", type(res_dict)
                if None not in res_dict.values():
                    print res_dict
                    rescount += 1
                elif with_empty:
                    print res_dict
            # print "Number of solutions found:", rescount
            print _info_msg[0] % rescount

    def solve(self, solve_inits=None, frozen={}, constraints=[], comp_eps=1e-4,
              system=None, select_expr=None, select_criterion=None):
        """
        Finds solution for determined or undetermined systems of equations using
        initial values of independent variables and frozen falues of excess variables.
 
        Parameters
        ----------
        solve_inits:      dict of form (variable: value (or array of values))
                          representing initial values of independent variables

        frozen:           dict of form (excess variable: value)

        constraints:      list of constraints used to filter solutions (Sympy types)

        comp_eps:         solution discrimination threshold (float)

        system:           equation system. Used to find values of additional variables
                          if they are present in solution celection expression

        select_expr:      expression used to select one of several possible solutions 
                          that satisfy constraints

        select_criterion: min or max - solution selection criterion. Used to select
                          best solution if several are found

        """

        # Prepare
        import solve
        reload(solve)
        solve._lambdified = self.lambdified
        criterion = self.selector_map.get(select_criterion, None)
        par_mix, sol_vars = self._mix_inits(solve_inits, len(frozen))
        
        # Solve
        reslist = []
        for vec in par_mix:
            sol_res = solve.fsolve(vec, sol_vars, frozen)
            reslist.append(Solution(frozen, sol_res))

        # Filter
        # reslist = self._compare_solutions(reslist, comp_eps)
        reslist = self._filter_solutions(reslist, constraints, comp_eps)

        if len(reslist) > 0 and criterion is not None:
            reslist = self._select_solution(reslist, select_expr, criterion, system)
 
        return reslist

    def evaluate(self, value_dict):
        """Evaluates function for a given values of variables"""
        import solve
        reload(solve)
        solve._lambdified = self.lambdified
        return solve.evaluate(value_dict)

    def sweep(self, sweep_vars, sweep_values, solve_inits=None, constraints=[], 
              system=None, select_expr=None, select_criterion=None,
              comp_eps=1e-4, verbose=True, multithread=False):
        """
        Parametric sweep. Finds solution for underdetermined systems based on 
        the input values of excess independent variables.

        Parameters
        ----------
        sweep_vars:       variables of parametric sweep

        sweep_values:     list (array) of values for sweep variables

        solve_inits:      dict of form (variable: value (or array of values))
                          representing initial values of independent variables

        constraints:      list of constraints used to filter solutions (Sympy types)

        system:           equation system. Used to find values of additional variables
                          if they are present in solution celection expression

        select_expr:      expression used to select one of several possible solutions 
                          that satisfy constraints

        select_criterion: min or max - solution selection criterion. Used to select
                          best solution if several are found

        comp_eps:         solution discrimination threshold (float)

        verbose:          whether or not to print info: 
                              1 or True - regular output (default)
                              2 - print extended output

        multithread:      bool, whether or not use multithreading
        """

        # Prepare
        self.sweep_vars = sweep_vars
        self.sweep_values = sweep_values
        swnum = len(sweep_vars)
        criterion = self.selector_map.get(select_criterion, None)
        par_mix, sol_vars = self._mix_inits(solve_inits, swnum)
        self.sw_grid = np.meshgrid(*sweep_values)
        self.sw_mix = np.array(self.sw_grid).T.reshape(-1, swnum)
        # self.sw_mix = np.array(np.meshgrid(*sweep_values)).T.reshape(-1, swnum)

        # Sweep
        if verbose:
            print "Sweep solve started..."
            t = time.time()

        import solve
        reload(solve)
        solve._lambdified = self.lambdified
        results = solve.sweep_solve(self.sw_mix, sweep_vars, par_mix, sol_vars, multithread)

        if verbose:
            # print "Sweep solve finished in %f seconds." % (time.time() - t)
            print _info_msg[1] % (time.time() - t)
        # warnings.filterwarnings("default")

        #Filter
        filtered_results = []
        for reslist in results:
            reslist = self._filter_solutions(reslist, constraints, comp_eps)

            if len(reslist) > 0 and criterion is not None:
                reslist = self._select_solution(reslist, select_expr, criterion, system)
            filtered_results.append(reslist)
        
        joinres = sum([reslist for reslist in filtered_results], [])
        self.results = joinres
        
        return joinres


    # Deprecated and probably broken
    def reduce(self, reducefunc, reducevar=None, resgrid=False, nonereplace=None): 
        """Returns value of the input function based on the found solution."""
        if reducevar is None:
            reducevar = sy.Symbol('reduced')

        reduced = []
        self.reduced_vals = []
        for res in self.results:
            if None not in res.values():
                rf = reducefunc.subs(res)
            else:
                # rf = nonereplace
                rf = None
            self.reduced_vals.append(rf)
            res_dict = {v: res[v] for v in self.sweep_vars}
            res_dict[reducevar] = rf
            reduced.append(res_dict)
        
        # if resgrid:
        #     z = np.asarray(self.reduced_vals, dtype=float)
        #     dims = map(len, self.sweep_values)
        #     z = z.reshape(tuple(dims))

            # try:
            #     z = z.reshape(tuple(dims))

            # except:
            #     warnings.warn("Some solutions are probably missing. Anable to reshape reduced values")

            # return self.sw_grid + [z]

        return reduced

    def reduced_grid(self, nonereplace=None):
        reduced_vals = []
        for redval in self.reduced_vals:
            if redval is None:
                reduced_vals.append(nonereplace)
            else:
                reduced_vals.append(redval)

        z = np.asarray(reduced_vals, dtype=float)
        dims = map(len, self.sweep_values)
        # print "dims:", dims
        z = z.reshape(tuple(reversed(dims))).T
        return self.sw_grid + [z]


    # _____________________________________________________________________

    # def _compare_solutions(self, reslist, comp_eps):
    #     compared = []
    #     tol_res = []
    #     hasnone = False
    #     for res in reslist:
    #         if None in res.values():
    #             if not hasnone:
    #                 compared.append(res)
    #                 hasnone = True
    #         else:
    #             # for val in res.values():
    #             #     print "resval_type:", type(val)
    #             res_ = Solution({var: float(val) for var, val in res.items()})
    #             res_int = [int(round(r/comp_eps)) for r in res_.values()]
    #             if res_int not in tol_res:
    #                 tol_res.append(res_int)
    #                 compared.append(res_)
    #     return compared

    # def _filter_solutions0(self, reslist, constraints):
    #     filtered = []
    #     for res in reslist:
    #         satisfy = True
    #         for constr in constraints:
    #             c = constr.subs(res)
    #             satisfy = satisfy & c
    #         if satisfy:
    #             filtered.append(res)
    #     return filtered