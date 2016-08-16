import math, pickle, warnings, time, json, copy
import sympy as sy
from solution import Solution

_error_msg = {
    0: ("System expects as maximum as 2 parameters which represent lists "
        "of objects and constraints"),
    1: "Equation %d cannot be solved against variable %s.",
    2: "All objects must have 'equations' attribute.",
    3: "At least one equation is required to construct system of equations.",
    4: "Arguments must be of type list or tuple."
}

_warn_msg = {
    
    0: ("Variable '%s' has been chosen as a redundant in equation #%d.\n" 
        "Call edit_sparse() method to replace it with another free variable "
        "in the equation."),
    1: "More then 1 solution found while replacing redundant variables.",
    2: "Keyword argument '%s' is not a valid argument and will be ignored."

}

_info_msg = {

    'head':  ("=======================================\n"
              "total number of variables:      %d\n"
              "total number of equations:      %d\n"
              "degrees of freedom:             %d\n"
              "=======================================\n"),

    'independent': "%d independent variables found: ",

    'redundant': "%d redundant variables found: ",

    'constants': "%d variables marked as constants: ",

    'mult': "Redundant variable '%s' has multiple solutions:\n",

    'mult_': ("\nBy default last solution has been chosen. "
              "Call replace_solution() method to specify other solution.\n"),

    'shrinked': "System has been shrinked by %d equations.\n",

    'shrinking': "\nShrinking system of equations...\n",

    'try_shrink': 'Try shrinking the system by calling shrink() method.',

}

class System(object):
    """
    Represents a system of equations.

    System of equations can be constructed from a list of equations, list of 
    objects that have attribute 'equations' (which is a list of equations) or both.

    Equations must be a SymPy objects of type Add, Mul or Pow.

    Parameters
    ----------
    objects:     list of objects that have attribute 'equations'. 

    constraints: list of equations.

    subs:        dictionary of variable substitutions.

    constants:   list of constants in equations.

    verbose:     controls the output - 0: silent mode; 1 or True: basic output,
                 2: extended output.

    Examples
    --------

    TODO

    """
    # def __init__(self, objects=[], constraints=[], subs={}, constants=[], verbose=True):
    def __init__(self, *args, **kwargs):

        valid_kwargs = ['subs', 'constants', 'verbose']
        # check_kwargs = [kwarg in valid_kwargs for kwarg in kwargs]
        for kwarg in kwargs:
            if kwarg not in valid_kwargs:
                warnings.warn(_warn_msg[2] % kwarg)

        subs = kwargs.get(valid_kwargs[0], {})
        self.constants = kwargs.get(valid_kwargs[1], [])
        self.verbose = kwargs.get(valid_kwargs[2], True)

        if len(args) > 2:
            raise ValueError(_error_msg[0])

        objects = []
        constraints = []
        oind = None
        cind = None
        constr_types = [sy.Add, sy.Mul, sy.Pow]
        for i, arg in enumerate(args):
            if type(arg) not in [list, tuple]:
                raise TypeError(_error_msg[4])
            if len(arg) > 0:
                check = [type(unit) in constr_types for unit in arg]
                if not check.count(False):
                    cind = i
                else:
                    try:
                        check_obj = [obj.equations for obj in objects]
                        oind = i
                    except:
                        raise ValueError(_error_msg[2])

        if oind is not None: objects = list(args[oind])
        if cind is not None: constraints = list(args[cind])

        # print "objects:", objects
        # print "constraints:", constraints

        if len(objects + constraints) == 0:
            raise ValueError(_error_msg[3])

        init_dictattr = ['redundant', 'sparse', 'multisols', 'sparse_edit']
        self.__dict__.update((name, {}) for name in init_dictattr)

        self.shrink_conv = 0
        self.long_varname = 0

        try:
            eq_list = sum([obj.equations.values() for obj in objects], constraints)
        except:
            eq_list = sum([obj.equations for obj in objects], constraints)
        self.equations = {i: eq.subs(subs) for i, eq in enumerate(eq_list)}
        self._copy_equations()

        # for i, eq in enumerate(eq_list):
        #     self.equations[i] = eq.subs(self.subs)
        #     varlist = list(eq.atoms(sy.Symbol))
        #     self.variables += varlist
        
        self.inspect()

    def _copy_equations(self):
        self.equations_tmp = copy.deepcopy(self.equations)
        self.tempmap = {i: i for i in self.equations}
    
    def _register_variables(self, add_varlist=[]):
        self.__dict__.update((str(v), v) for v in self.variables)
        # self.__dict__.update((str(v), v) for v, s in self.sparse.values())
        self.__dict__.update((str(v), v) for v in self.redundant)
        self.__dict__.update((str(v), v) for v in self.constants)
        self.__dict__.update((str(v), v) for v in add_varlist)

        # print "self.__dict__:"
        # for v in self.__dict__:
        #     print v
        # print

    def _cleanup_constants(self):
        const_set = set(self.constants)
        muddy_constants = const_set - self.variables
        self.constants = list(const_set - muddy_constants)

    def _sort_equations(self):
        # self.equations = {i: eq for i, eq in enumerate(self.equations.values())}
        # self.equations = {}
        equations = {}
        for j, (i, eq) in enumerate(self.equations.items()):
            equations[j] = eq
            self.tempmap[j] = i
        self.equations = equations
        # self.tempmap

    def _add_sparse(self, i, var, free_list):
        eq = self.equations[i]
        self.sparse[i] = (var, eq)
        free_list.append(var)

    def _delve_nonsparse(self, nonsparse_map, free_list):
        equations_ = copy.deepcopy(self.equations)
        for i, free_vars in nonsparse_map.items():
            redundant = None
            eq_temp = equations_[i]

            # retrieve set of variables in all equations except the current one
            del equations_[i]
            variables_set = set(sum((list(eq.free_symbols) for eq in equations_.values()), []))
            equations_[i] = eq_temp

            # select variables that are not present in free_list
            del nonsparse_map[i]
            vars_selected = []
            free_set = set(free_list)
            for var in free_vars:
                if var not in free_set:
                    vars_selected.append(var)

            if len(vars_selected) == 1:
                redundant = vars_selected[0]
            elif len(vars_selected) > 1:
                # select unique variables that are not present in any other equation 
                var_stack = set(sum(nonsparse_map.values(), list(variables_set)))
                vars_selected_ = []
                for var in free_vars:
                    if var not in var_stack:
                        vars_selected_.append(var)

                if len(vars_selected_) == 0:
                    redundant = vars_selected[0]
                    # warnmsg = _warn_msg[0] % (str(var), i)
                    warnmsg = _warn_msg[0] % (str(redundant), i)
                    warnings.warn(warnmsg)
                elif len(vars_selected_) == 1:
                    redundant = vars_selected_[0]
            if redundant is not None:
                self._add_sparse(i, redundant, free_list)
            nonsparse_map[i] = free_vars

    def _extract_sparse(self, deep=False):
        free_list = []
        nonsparse_map = {}
        for i, eq in self.equations.items():
            free_vars = self._find_free(eq)
            mapped = [ v[0] for v in self.sparse.values() ]
            if len(free_vars) == 1 and free_vars[0] not in mapped:
                self._add_sparse(i, free_vars[0], free_list)
            elif len(free_vars) > 1:
                nonsparse_map[i] = free_vars

        if deep:
            self._delve_nonsparse(nonsparse_map, free_list)
        return list(set(free_list))

    def _make_summary(self, verbose):
        self.dof = len(self.variables) - len(self.equations) - len(self.constants)
        self.independent = self.variables - set(self.free_list + self.constants)
        redundnum = len(self.free_list)
        if verbose:
            varnum = len(self.variables) - len(self.constants)
            info_head = _info_msg['head'] % (varnum, len(self.equations), self.dof)
            info_independent = _info_msg['independent'] % len(self.independent)
            info_redundant = _info_msg['redundant'] % redundnum
            info_constants = _info_msg['constants'] % len(self.constants)
            summary = (info_head + info_independent + str(list(self.independent)) +
                       '\n' + info_redundant + str(self.free_list) + '\n' +
                       info_constants + str(list(self.constants)))
            print summary
            self.print_equations()
            if verbose == 2 and len(self.sparse) > 0:
                self.print_sparse()
            if (redundnum > 0) and (len(self.equations) > 1):
                print _info_msg['try_shrink']

    def _solve(self, eq, var):
        c = eq.coeff(var)
        try:
            cf = float(c)
            extracted = eq - c*var
            if var not in list(extracted.free_symbols):
                return extracted/abs(c)
        except: pass
            # warnings.filterwarnings("always")
        # print "var:", var
        # print "equation:", eq
        sol = sy.solve(eq, var)
        if len(sol) > 1:
            self.multisols[var] = sol
            warnings.warn(_warn_msg[1])
            print _info_msg['mult'] % str(var), sol, _info_msg['mult_']
        return sol[-1]
            # return sy.solve(eq, var)

    def _compress(self):
        if self.verbose:
            print "Compressing..."
        # find equations with one free dependent variable  
        replaced = []
        min_varset_len = 0
        varsets = {}
        j = 0
        for i, (var, eq) in self.sparse.items():
            varset = eq.free_symbols - set(self.constants + [var]) - self.independent
            varset_len = len(varset)
            min_varset_len = varset_len if j == 0 else min(varset_len, min_varset_len)
            varsets[i] = varset
            j += 1

        for i, (var, eq) in self.sparse.items():
            varset = varsets[i]
            if (len(varset) <= min_varset_len) and (len(self.equations) > 1):
            # if len(varset) == 0:
            # if (len(varset) <= min_varset_len):
                self.redundant[var] = self._solve(eq, var)
                replaced.append(var)
                del self.equations[i]
                del self.sparse[i]  

        self._update()  
        if self.verbose:
            print "Replaced variables:", replaced, "\n"
            # print

        if self.verbose == 2:
            self.print_sparse()
            self.print_replacementset()

    def _update(self):
        # self.equations = self.equations_tmp

        # update system of equations
        self.equations = {i: eq.subs(self.redundant) 
                          for i, eq in self.equations.items()}
        
        # update sparse equations
        self.sparse = {i: (var, eq.subs(self.redundant)) 
                       for i, (var, eq) in self.sparse.items()}
        
        # find new sparse equations
        self._extract_sparse()


    def _find_free(self, eq):
        """returns a list of free variables in the equation 
           which can be easily extracted"""
        # adds = list(eq.atoms(sy.Add))
        const_to_ones = {const: 1 for const in self.constants}
        eqq = eq.subs(const_to_ones)

        # muls = list(eq.atoms(sy.Mul))
        # mulns = list((eq*(-1)).atoms(sy.Mul))
        # pows = list(eq.atoms(sy.Pow))

        # muls = list(eqq.atoms(sy.Mul))
        # mulns = list((eqq*(-1)).atoms(sy.Mul))
        # pows = list(eqq.atoms(sy.Pow))

        # print "eq:", eq
        # # print "adds:", adds
        # print "muls:", muls
        # print "mulns:", mulns
        # print "pows:", pows
        # # print

        # break equation into bricks
        bricks = map(eqq.atoms, [sy.Mul, sy.Pow]) + [(eqq*(-1)).atoms(sy.Mul)]
        bricks = sum(map(list, bricks), [])
        # print "bricks:\n", bricks
        # print "(muls + mulns + pows):\n", (muls + mulns + pows)
        # print

        var_combs = [list(atom.free_symbols) for atom in bricks]
        unique_combs = {sum(map(ord, str(x))): x for x in var_combs}.values()
        # print "unique_combs:", unique_combs
        occurances = {var: 0 for var in self.variables}
        const_set = set(self.constants)

        singles = set()
        for comb in unique_combs:
            for var in comb: occurances[var] += 1
            # if len(comb) == 1: singles.append(comb.pop())
            if len(comb) == 1: singles.add(comb.pop())

        # print "singles:", singles
        free = []
        for var in singles:
            if occurances[var] == 1: free.append(var)
        # print "occurances:", occurances

        return list(set(free) - set(self.constants))

    def _find_adopt_order(self, redundant, sol_vars, converged):
        convars = []
        sol_vars_set = set(sol_vars)
        for var, eq in redundant.items():
            variables = eq.free_symbols - sol_vars_set
            if len(variables) == 0:
                converged[var] = [s for s in sol_vars]
                convars.append(var)
                del redundant[var]
                sol_vars.append(var)
        return convars

    def _adopt_order(self):
        from_sparse = [sp[0] for sp in self.sparse.values()]
        sol_vars = list(self.independent) + from_sparse
        redundant = copy.deepcopy(self.redundant)
        self.adopt_order = []
        self.converged = {}
        for v in self.redundant:
            convars = self._find_adopt_order(redundant, sol_vars, self.converged)
            if len(convars) == 0:
                break
            else:
                self.adopt_order.append(convars)

    def lambdify_redundant(self):
        self._adopt_order()
        self.lambdified = {}
        for var, variables in self.converged.items():
            eq = self.redundant[var]
            self.lambdified[var] = [sy.lambdify(variables, eq), variables]

    def adopt(self, result, verbose=False):
        """Adopts solution and calculates values of all redundant variables."""

        if not hasattr(self, 'lambdified'):
            self.lambdify_redundant()

        if type(result) in [list, tuple]:
            reslist = result
        elif type(result) in (Solution, dict):
            reslist = [result]
        else:
            raise ValueError("adoptdict type must represent solution or list of solutions")

        adopted = []
        for res in reslist:

            if None in res.values():
                nonedict = {var: None for var in self.redundant}
                res_dict = Solution(res, nonedict)
            else:
                res_extend = copy.deepcopy(res)
                for convars in self.adopt_order:
                    for var in convars:
                        func, variables = self.lambdified[var]
                        values = [res_extend[v] for v in variables]
                        try:
                            res_extend[var] = func(*values)
                        except:
                            # lambdified doesn't handle complex output
                            subsres = self.redundant[var].subs(res_extend)

                            converted = complex(subsres.evalf())
                            # print "got complex"
                            if converted.imag == 0:
                                res_extend[var] = converted.real
                            else:
                                res_extend[var] = converted
                res_dict = Solution(res_extend)

            adopted.append(res_dict)
            if verbose: print res_dict
        return adopted

    def inspect(self):
        variables = [ list(eq.atoms(sy.Symbol)) for eq in self.equations.values() ]
        self.variables = set(sum(variables, []))
        self.long_varname = max(map(len, [str(var) for var in self.variables]))
        self._cleanup_constants()
        self.free_list = self._extract_sparse(deep=True)
        self._register_variables()
        self._make_summary(self.verbose)

    def shrink(self, simplify=False):
        # self.equations_tmp = copy.deepcopy(self.equations)
        self._copy_equations()
        if self.verbose:
            print _info_msg['shrinking']
        extreme = len(self.sparse)
        _shrinks = [extreme]
        for i in range(extreme):
            if self.verbose:
                print "Iteration %d:" %i
            self._compress()
            size = len(self.sparse)
            _shrinks.append(size)
            if (_shrinks[i+1] == _shrinks[i]) or (size == 0):
                break

        if self.verbose:
            print _info_msg['shrinked'] % (extreme - len(self.sparse))

        self._sort_equations()
        if simplify: self.try_eliminate()
        self.inspect()

        # conv = abs(self.redundnum - self.shrink_conv)
        # if self.redundnum > 0 and conv!=0:
        #     self.shrink_conv = self.redundnum
        #     self.shrink()

    def subs_safe(self, subs_dict, verbose=False):
        equations = {i: eq.subs(subs_dict, simultaneous=True)
                     for i, eq in self.equations.items()}
        if verbose: self.print_equations(equations)
        return equations

    def subs(self, subs_dict, simplify=False, constants=[]):
        # for i, var in subs_dict:
        # print "self.constants before:", self.constants
        self.constants += constants
        # t = 0
        for var in subs_dict:
            if var in self.constants:
                i = self.constants.index(var)
                # print "i, var:", i, var
                self.constants.pop(i)
                # t += 1
                # if type(var) is sy.Symbol:
                #     self.constants.append(var)
        # print "self.constants after:", self.constants
        sp = lambda eq: sy.simplify(eq) if simplify else eq 
        self.equations = {i: sp(eq.subs(subs_dict, simultaneous=True))
                          for i, eq in self.equations.items()}

        self.redundant = {var: eq.subs(subs_dict, simultaneous=True) 
                          for var, eq in self.redundant.items()}

        self.sparse = {i: (var, eq.subs(subs_dict, simultaneous=True)) 
                        for i, (var, eq) in self.sparse.items()}                   
        self.inspect()
        # print "self.variables:", self.variables
        
        new_variables = []
        for eq in subs_dict.values():
            try:
                new_variables += eq.free_symbols
            except:
                pass
        self._register_variables(new_variables)

    def try_eliminate(self):
        """can be used to eliminate unnecessary variables"""
        self.equations = {i: sy.simplify(sy.together(eq))
                  for i, eq in self.equations.items()}

    def edit_sparse(self, edit_dict, verbose=False):
        for i, var in edit_dict.items():
            var_, eq = self.sparse[i]
            free_vars = self._find_free(eq)
            if var in free_vars:
                self.sparse_edit = {i: var}
                self.sparse[i] = (var, self.equations[i])
                self.free_list = list(set(self.free_list) - set([var_])) + [var]
                # self._make_summary()
            else:
                raise ValueError(_error_msg[1] % (i, str(var)))
        self._make_summary(False)
        if verbose and len(self.sparse) > 0:
            self.print_sparse()

    def replace_solution(self, replace_dict):
        for var, solnum in replace_dict.items():
            sol = self.multisols[var][solnum]
            self.redundant[var] = sol
        
        self.equations = {i: self.equations_tmp[self.tempmap[i]] 
                          for i in self.equations}
        self._update()
        self.print_replacementset()
        self.inspect()

    def print_sparse(self):
        msg = 'Sparse equations:\n'
        print (msg + '-'*len(msg))
        pvar = 'redundant: %s'
        # w = max(map(len, [ str(var) for var in self.variables ])) + len(p)
        w = self.long_varname + len(pvar)
        for i, (var, eq) in self.sparse.items():
            rs = pvar % str(var)
            w_ = w - len(rs) + 2
            print rs + ' ' * w_ + '[%d]:' % i, eq
        print

    def print_replacementset(self):
        msg = "Replacement set:\n"
        print (msg + '-'*len(msg))         
        for var, eq in self.redundant.items():
            svar = str(var)
            pvar = '[%s]:' + ' '*(self.long_varname - len(svar) + 1)
            print pvar % svar, eq
        print

    def print_equations(self, equations=None):
        if equations is None:
            equations = self.equations
        msg = "\nEquations:\n"
        print (msg + '-'*len(msg))
        for i, eq in equations.items():
            # print "[%d]:  " % (i+1), eq
            print "[%d]:  " % i, eq
        print





    # def _adopt(self, res, nonnumeric, extended): # REMOVE IT
    #     for var, eq in res.items():
    #         # res[var] = eq.subs(extended, simultaneous=True).evalf()
    #         res[var] = eq.subs(extended)
    #         try:
    #             f = float(res[var])
    #             extended[var] = f
    #             del nonnumeric[var]
    #             # nonnumeric.pop()
    #         except: 
    #             # nonnumeric[var] = eq
    #             pass

    # def _adopt2(self, redundant, nonnumeric, res, converged):
    #     # converged = {}
    #     convars = []
    #     sol_vars = res.keys()
    #     for var, eq in redundant.items():
    #         variables = sol_vars + list(eq.free_symbols)
    #         # res[var] = eq.subs(extended, simultaneous=True).evalf()
    #         redundant[var] = eq.subs(res).evalf()
    #         try:
    #             f = complex(redundant[var])
    #             print "got complex"
    #             if f.imag == 0:
    #                 res[var] = f.real
    #             else:
    #                 res[var] = f
    #             # del nonnumeric[var]
    #             del redundant[var]
    #             converged[var] = variables
    #             convars.append(var)
    #             # nonnumeric.pop()
    #         except: 
    #             # nonnumeric[var] = eq
    #             pass
    #     return convars

    # def adopt(self, result, constdict={}, verbose=False, verify_subs=False):  # REMOVE IT
    #     """Adopts solution and calculates values of all redundant variables."""

    #     if type(result) in [list, tuple]:
    #         reslist = result
    #     elif type(result) in (Solution, dict):
    #         reslist = [result]
    #     else:
    #         raise ValueError("adoptdict type must represent solution or list of solutions")

    #     adopted = []
    #     for res in reslist:

    #         if None in res.values():
    #             nonedict = {var: None for var in self.redundant}
    #             res_dict = Solution(res, nonedict)
    #             # pass
    #         else:
    #             # nonnumeric = copy.deepcopy(self.redundant)
    #             nonnumeric = {var: 1 for var in self.redundant}
    #             # nonnumeric = range(len(self.redundant))
    #             # extended = dict(res, **constdict)
    #             extended = res
    #             tosolve = copy.deepcopy(self.redundant)

    #             for i in range(len(tosolve)):
    #                 self._adopt(tosolve, nonnumeric, extended)
    #                 if len(nonnumeric) == 0:
    #                     # print "break", len(nonnumeric), len(self.redundant)
    #                     break

    #             # print "extended after:", extended
    #             # for var, rr in tosolve.items():
    #             #     print "[%s]:" %str(var), rr
                
    #             res_dict = Solution(res, tosolve)

    #         adopted.append(res_dict)
    #         if verbose: print res_dict
    #             # return res_dict
    #     return adopted