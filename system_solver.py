import math, pickle, warnings, time, json, copy
import multiprocessing as mp
import sympy as sy
import numpy as np
import scipy.optimize as so
from solution import Solution

_error_msg = {
    0: 'init_vec parameter must represent initial values of variables'
}

def _sweepsolve(input_list):
    sweeps, sweep_vars, sol_vars, par_mix, comp_eps = input_list
    reslist = []
    sweep_dict = {ss: sv for ss, sv in zip(sweep_vars, sweeps)}
    tol_res = []
    for vec in par_mix:
        try:
            # res = so.fsolve(_evaluate_lambdified, vec, args=(sweep_dict, sol_vars, _lambdified))
            res = so.fsolve(_evaluate_wrapper, vec, args=(sweep_dict, sol_vars, _lambdified))
            res_int = [int(round(r/comp_eps)) for r in res]
            if res_int not in tol_res:
                tol_res.append(res_int)
                sol_dict = {v: r for v, r in zip(sol_vars, res)}
                # res_dict = dict(sweep_dict, **sol_dict)
                # res_dict = Solution(sweep_dict, **sol_dict)
                res_dict = Solution(sweep_dict, sol_dict)
                reslist.append(res_dict)
        except: #RuntimeWarning
            # pass
            sol_dict = {v: None for v in sol_vars}
        
            # res_dict = dict(sweep_dict, **sol_dict)
            # res_dict = Solution(sweep_dict, **sol_dict)
            res_dict = Solution(sweep_dict, sol_dict)
            reslist.append(res_dict)
            # pass
    return reslist

# def _evaluate_lambdified(values, sweep_dict, sol_vars, lambdified):
#     values = map(float, values)
#     value_dict = dict({s:v for s, v in zip(sol_vars, values)}, **sweep_dict)

#     # warnings.filterwarnings("error")
#     result = []
#     for func, param_str in lambdified:
#         args = [value_dict[varstr] for varstr in param_str]
#         # print "args:", args
#         try:
#             res = func(*args)
#         except:
#             res = 1e10
#         result.append(res)
#     if len(value_dict) == 1:
#         return result[0]
#     else:
#         return result

def _evaluate(value_dict, lambdified):
    # warnings.filterwarnings("error")
    result = []
    for func, param_str in lambdified:
        args = [value_dict[varstr] for varstr in param_str]
        # print "args:", args
        try:
            res = func(*args)
        except:
            res = 1e10
        result.append(res)
    if len(value_dict) == 1:
        return result[0]
    else:
        return result

def _evaluate_wrapper(values, sweep_dict, sol_vars, lambdified):
    values = map(float, values)
    value_dict = dict({s:v for s, v in zip(sol_vars, values)}, **sweep_dict)
    return _evaluate(value_dict, lambdified)

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

        global _lambdified
        self.equations = equations
        self.variables = []
        self.lambdified = []
        for i, eq in self.equations.items():
            variables = eq.free_symbols
            self.lambdified.append([sy.lambdify(variables, eq), variables])
            self.variables += variables
        self.variables = set(self.variables)
        _lambdified = self.lambdified

    def _evaluate(self, values, sweep_dict):
        return _evaluate_wrapper(values, sweep_dict, self.sol_vars, self.lambdified)

    def _filter_solutions(self, reslist, constraints):
        filtered = []
        # hasnone = False
        for res in reslist:
            # res_dict = {v:r for v, r in zip(variables, res)}
            # res_dict = {v:r for v, r in zip(self.sol_vars, res)}
            if None in res.values():
                filtered.append(res)
            else:
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

    def _sweep_singlecore(self):
        results = []
        for sweeps in self.sw_mix:
            # sweep_dict = {ss: sv for ss, sv in zip(self.sweep_vars, sweeps)}
            # res = _sweepsolve(sweep_dict, self.sol_vars, par_mix, self.lambdified, comp_eps)
            input_list = [sweeps, self.sweep_vars, self.sol_vars, self.par_mix, self.comp_eps]
            res = _sweepsolve(input_list)

            # print "res:", res
            results.append(res)
        return results

    def _sweep_multicore(self):
        corenums = mp.cpu_count()
        if corenums > 1: corenums -= 1
        pool = mp.Pool(processes=corenums)
        # joblist = [(sweeps, deepcopy(self)) for sweeps in sw_mix]
        statargs = [self.sweep_vars, self.sol_vars, self.par_mix, self.comp_eps]
        joblist = [[sweeps] + statargs for sweeps in self.sw_mix]
        mapresult = pool.map_async(_sweepsolve, joblist)
        pool.close()
        pool.join()

        results = []
        for reslist in mapresult.get():
            results.append(reslist)
        return results

    def show_res(self):
        if hasattr(self, "results"):
            rescount = 0
            for res_dict in self.results:
                # print "res type:", type(res_dict)
                if None not in res_dict.values():
                    print res_dict
                    rescount += 1
            print "Number of solutions found:", rescount

    def solve(self, solve_inits=None, constants={}, constraints=[]):
        """Finds solution for determined system of equations."""
        warnings.filterwarnings("always")

        if solve_inits is None:
            init_values = [0.0]*paramnum
            self.sol_vars = self.variables
        else:
            vs = [(k, v) for k, v in solve_inits.items()]
            self.sol_vars, init_values = zip(*vs)
        warnings.filterwarnings("error")
        # try:
        res = so.fsolve(self._evaluate, init_values, args=(constants,))
        res_dict = {v: res[i] for i, v in enumerate(self.sol_vars)}
        res_nones = {v: None for v in res_dict}

            # res_dict = dict(constants, **res_dict)
        # except:
        #     warnings.filterwarnings("default")
        #     warnings.warn("Solution hasn't been found")
        #     res = []
        #     res_dict = {v: None for v in self.sol_vars}
        # res_dict = dict(constants, **res_dict)
        # res_dict = Solution(constants, **res_dict)
        res_dict = Solution(constants, res_dict)

        satisfy = True
        for constr in constraints:
            c = constr.subs(res_dict)
            satisfy = satisfy & c
        if satisfy:
            return res_dict
        else:
            return Solution(constants, res_nones)

    def evaluate(self, value_dict):
        return _evaluate(value_dict, self.lambdified)

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
              minfunc=None, maxfunc=None, comp_eps=1e-4, verbose=True):
        """
        Parametric sweep. Finds solution for underdetermined systems based on 
        the input values of excess independent variables.
        """

        self.comp_eps = comp_eps
        self.sweep_vars = sweep_vars
        self.sweep_values = sweep_values
        swnum = len(sweep_vars)
        paramnum = len(self.variables) - swnum

        m = np.nonzero([minfunc is not None, maxfunc is not None])[0]
        if len(m) > 0:
            select_func, criterion = [(minfunc, 'min'), (maxfunc, 'max')][m[0]]

        if solve_inits is None:
            self.par_mix = [[0.0]*paramnum]
            self.sol_vars = self.variables - set(sweep_vars)
        elif len(solve_inits) == paramnum:
            vs = [(k, v) for k, v in solve_inits.items()]
            self.sol_vars, init_values = zip(*vs)
            self.par_mix = np.array(np.meshgrid(*init_values)).T.reshape(-1, paramnum)
        else:
            raise ValueError(_error_msg[0])
        self.sw_grid = np.meshgrid(*sweep_values)
        self.sw_mix = np.array(self.sw_grid).T.reshape(-1, swnum)
        # self.sw_mix = np.array(np.meshgrid(*sweep_values)).T.reshape(-1, swnum)

        warnings.filterwarnings("always")
        warnings.filterwarnings("error")
        if verbose:
            print "Sweep solve started..."
            t = time.time()
        # results = self._sweep_multicore()
        results = self._sweep_singlecore()
        if verbose:
            print "Sweep solve finished in %f seconds." % (time.time() - t)
        warnings.filterwarnings("default")

        filtered_results = []
        for reslist in results:
            if len(constraints) > 0:
                reslist = self._filter_solutions(reslist, constraints)
            if len(reslist) > 0 and len(m) > 0:
                reslist = self._select_solution(reslist, select_func, criterion)
            filtered_results.append(reslist)
        
        # filtered_results = results
        # print "filtered_results:", filtered_results
        joinres = sum([reslist for reslist in filtered_results], [])
        self.results = joinres
        
        return joinres

