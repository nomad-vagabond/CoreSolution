import sys, os
import math, pickle, warnings, time, json, copy
import multiprocessing as mp
import sympy as sy
import numpy as np
import scipy.optimize as so
from solution import Solution

_error_msg = {
    0: 'init_vec parameter must represent initial values of variables'
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

    def _filter_solutions(self, reslist, constraints, comp_eps, include_nones=True):
        filtered = []
        # hasnone = False
        tol_res = []
        for res in reslist:
            # res_dict = {v:r for v, r in zip(variables, res)}
            # res_dict = {v:r for v, r in zip(self.sol_vars, res)}
            if None in res.values():
                if include_nones:
                    filtered.append(res)
            else:
                # print "res:", res
                res_int = [int(round(r/comp_eps)) for r in res.values()]
                if res_int not in tol_res:
                    tol_res.append(res_int)
                    satisfy = True
                    for constr in constraints:
                        # c = const.subs(res_dict)
                        c = constr.subs(res)
                        satisfy = satisfy & c
                    if satisfy:
                        filtered.append(res)
        
        # print "solutions:", solutions
        # print "filtered:", filtered
        return filtered

    def _select_solution(self, reslist, expression, criterion):
        # variables = sy.symbols(self.vals_str)
        target = []
        hasnone = False
        nonempty = []

        for res in reslist:
            if None in res.values(): 
                hasnone = True
            else:
                subs_exp = expression.subs(res)
                target.append(subs_exp.evalf())
                nonempty.append(res)

        if len(target) > 0:
            exec 'i = np.arg%s(target)' % criterion
            return [nonempty[i]]
        elif hasnone:
            return [reslist[0]]

    def _mix_inits(self, solve_inits, swnum):
        paramnum = len(self.variables) - swnum

        if solve_inits is None:
            # init_values = [[0.0]*len(self.variables)]
            par_mix = [[0.0]*paramnum]
            sol_vars = self.variables
        elif len(solve_inits) == paramnum:
            vs = [(k, v) for k, v in solve_inits.items()]
            sol_vars, init_values = zip(*vs)
            par_mix = np.array(np.meshgrid(*init_values)).T.reshape(-1, paramnum)
        else:
            raise ValueError(_error_msg[0])

        return par_mix, sol_vars

    def show_res(self):
        if hasattr(self, "results"):
            rescount = 0
            for res_dict in self.results:
                # print "res type:", type(res_dict)
                if None not in res_dict.values():
                    print res_dict
                    rescount += 1
            print "Number of solutions found:", rescount

    def solve(self, solve_inits=None, frozen={}, constraints=[], comp_eps=1e-4):
        """Finds solution for determined system of equations."""

        import solve
        reload(solve)
        solve._lambdified = self.lambdified

        par_mix, sol_vars = self._mix_inits(solve_inits, len(frozen))
        
        reslist = []
        for vec in par_mix:
            sol_res = solve.fsolve(vec, sol_vars, frozen)
            reslist.append(Solution(frozen, sol_res))
        # inputlist = [sweep_values, sweep_vars, sol_inits, sol_vars]
        # res = _find_solutions(inputlist)

        if len(constraints) > 0:
            # print "comp_eps:", comp_eps
            reslist = self._filter_solutions(reslist, constraints, comp_eps, include_nones=False)

        # if len(reslist) > 0 and len(m) > 0:
        #     reslist = self._select_solution(reslist, select_func, criterion)
 

        return reslist


        # sol_res = solve.fsolve(init_values, sol_vars, frozen, solve_order=self.solve_order)
        # res_nones = {v: None for v in sol_res}
        # res_dict = Solution(frozen, sol_res)

        # if None in sol_res.values():
        #     satisfy = True
        #     for constr in constraints:
        #         c = constr.subs(res_dict)
        #         satisfy = satisfy & c
        #     if not satisfy:
        #         res_dict = Solution(frozen, res_nones)
        #     # else:
        # return res_dict

    def evaluate(self, value_dict):
        """Evaluates function for a given values of variables"""
        import solve
        reload(solve)
        solve._lambdified = self.lambdified
        return solve.evaluate(value_dict)

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

    def sweep(self, sweep_vars, sweep_values, solve_inits=None, constraints=[], 
              minfunc=None, maxfunc=None, comp_eps=1e-4, verbose=True, 
              multithread=False, adopt_syst=None):
        """
        Parametric sweep. Finds solution for underdetermined systems based on 
        the input values of excess independent variables.

        Parameters
        ----------
        sweep_vars:   variables of parametric sweep
        sweep_values: list (array) of values for sweep variables
        solve_inits:  dict of form (variable: value (or array of values)) representing
                      initial values of independent variables
        constraints:  list of constraints used to filter solutions (Sympy types)
        minfunc:      minimization function. Used to select proper better solution if 
                      more then one solution is found
        maxfunc:      maximization function. Used to select proper better solution if 
                      more then one solution is found
        comp_eps:     solution discrimination threshold (float)
        verbose:      bool or int, whether or not to print info: 
                        1 or True - regular output (default)
                        2 - print extended output
        multithread:  bool, whether or not use multithreading
        """

        # self.comp_eps = comp_eps
        self.sweep_vars = sweep_vars
        self.sweep_values = sweep_values
        swnum = len(sweep_vars)
        # paramnum = len(self.variables) - swnum

        m = np.nonzero([minfunc is not None, maxfunc is not None])[0]
        if len(m) > 0:
            select_func, criterion = [(minfunc, 'min'), (maxfunc, 'max')][m[0]]

        # if solve_inits is None:
        #     par_mix = [[0.0]*paramnum]
        #     self.sol_vars = self.variables - set(sweep_vars)
        # elif len(solve_inits) == paramnum:
        #     vs = [(k, v) for k, v in solve_inits.items()]
        #     self.sol_vars, init_values = zip(*vs)
        #     par_mix = np.array(np.meshgrid(*init_values)).T.reshape(-1, paramnum)
        # else:
        #     raise ValueError(_error_msg[0])


        par_mix, sol_vars = self._mix_inits(solve_inits, swnum)

        self.sw_grid = np.meshgrid(*sweep_values)
        self.sw_mix = np.array(self.sw_grid).T.reshape(-1, swnum)
        # self.sw_mix = np.array(np.meshgrid(*sweep_values)).T.reshape(-1, swnum)

        # warnings.filterwarnings("always")
        # warnings.filterwarnings("error")
        if verbose:
            print "Sweep solve started..."
            t = time.time()

        import solve
        reload(solve)
        solve._lambdified = self.lambdified

        # results = solve.sweep_solve(self.sw_mix, sweep_vars, par_mix, self.sol_vars, multithread)
        results = solve.sweep_solve(self.sw_mix, sweep_vars, par_mix, sol_vars, multithread)

        if verbose:
            print "Sweep solve finished in %f seconds." % (time.time() - t)
        # warnings.filterwarnings("default")

        filtered_results = []
        for reslist in results:
            if len(constraints) > 0:
                # print "comp_eps:", comp_eps
                reslist = self._filter_solutions(reslist, constraints, comp_eps)
            if len(reslist) > 0 and len(m) > 0:
                reslist = self._select_solution(reslist, select_func, criterion)
            filtered_results.append(reslist)
        
        # filtered_results = results
        # print "filtered_results:", filtered_results
        joinres = sum([reslist for reslist in filtered_results], [])
        self.results = joinres
        
        return joinres

