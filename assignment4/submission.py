from __future__ import print_function

import collections, util, copy

############################################################
# Problem 3.1a
import queue


def create_nqueens_csp(n = 8):
    """
    Return an N-Queen problem on the board of size |n| * |n|.
    You should call csp.add_variable() and csp.add_binary_potential().

    @param n: number of queens, or the size of one dimension of the board.

    @return csp: A CSP problem with correctly configured potentials
        such that it can be solved by a weighted CSP solver.
    """
    csp = util.CSP()
    # BEGIN_YOUR_CODE (around 7 lines of code expected)
    variables = ["x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7", "x_8"]
    for var in variables:
        csp.add_variable(var, [i+1 for i in range(8)])

    for i, x_i in enumerate(variables):
        for j, x_j in enumerate(variables):
            if i != j:
                csp.add_binary_potential(x_i, x_j, lambda x, y: x != y and abs(x-y) != abs(i-j))

    # raise Exception("Not implemented yet")
    # END_YOUR_CODE
    return csp


# create_nqueens_csp()
############################################################
# Problem 3.1b

# A backtracking algorithm that solves weighted CSP.
# Usage:
#   search = BacktrackingSearch()
#   search.solve(csp)
class BacktrackingSearch():

    def reset_results(self):
        """
        This function resets the statistics of the different aspects of the
        CSP sovler. We will be using the values here for grading, so please
        do not make any modification to these variables.
        """
        # Keep track of the best assignment and weight found.
        self.optimalAssignment = {}
        self.optimalWeight = 0

        # Keep track of the number of optimal assignments and assignments. These
        # two values should be identical when the CSP is unweighted or only has binary
        # weights.
        self.numOptimalAssignments = 0
        self.numAssignments = 0

        # Keep track of the number of times backtrack() gets called.
        self.numOperations = 0

        # Keep track of the number of operations to get to the very first successful
        # assignment (doesn't have to be optimal).
        self.firstAssignmentNumOperations = 0

        # List of all solutions found.
        self.allAssignments = [] 
        self.allWeights = []

    def print_stats(self):
        """
        Prints a message summarizing the outcome of the solver.
        """
        if self.optimalAssignment:
            print ("Found %d optimal assignments with weight %f in %d operations" % \
                (self.numOptimalAssignments, self.optimalWeight, self.numOperations))
            print ("First assignment took %d operations" % self.firstAssignmentNumOperations)
        else:
            print ("No solution was found.")

    def get_delta_weight(self, assignment, var, val):
        """
        Given a CSP, a partial assignment, and a proposed new value for a variable,
        return the change of weights after assigning the variable with the proposed
        value.

        @param assignment: A list of current assignment. len(assignment) should
            equal to self.csp.numVars. Unassigned variables have None values, while an
            assigned variable has the index of the value with respect to its
            domain. e.g. if the domain of the first variable is [5,6], and 6
            was assigned to it, then assignment[0] == 1.
        @param var: Index of an unassigned variable.
        @param val: Index of the proposed value with resepct to |var|'s domain.

        @return w: Change in weights as a result of the proposed assignment. This
            will be used as a multiplier on the current weight.
        """
        assert assignment[var] is None
        w = 1.0
        if self.csp.unaryPotentials[var]:
            w *= self.csp.unaryPotentials[var][val]
            if w == 0: return w
        for var2, potential in self.csp.binaryPotentials[var].items():
            if assignment[var2] == None:
                continue  # Not assigned yet
            w *= potential[val][assignment[var2]]
            if w == 0:
                return w
        return w

    def solve(self, csp, mcv = False, lcv = False, mac = False):
        """
        Solves the given weighted CSP using heuristics as specified in the
        parameter. Note that unlike a typical unweighted CSP where the search
        terminates when one solution is found, we want this function to find
        all possible assignments. The results are stored in the variables
        described in reset_result().

        @param csp: A weighted CSP.
        @param mcv: When enabled, Most Constrained Variable heuristics is used.
        @param lcv: When enabled, Least Constraining Value heuristics is used.
        @param mac: When enabled, AC-3 will be used after each assignment of an
            variable is made.
        """
        # CSP to be solved.
        self.csp = csp

        # Set the search heuristics requested asked.
        self.mcv = mcv
        self.lcv = lcv
        self.mac = mac

        # Reset solutions from previous search.
        self.reset_results()

        # The list of domains of every variable in the CSP. Note that we only
        # use the indices of the values. That is, if the domain of a variable
        # A is [2,3,5], then here, it will be stored as [0,1,2]. Original domain
        # name/value can be obtained from self.csp.valNames[A]
        self.domains = [list(range(len(domain))) for domain in self.csp.valNames]
        # Perform backtracking search.
        self.backtrack([None] * self.csp.numVars, 0, 1)

        # Print summary of solutions.
        self.print_stats()

    def backtrack(self, assignment, numAssigned, weight):
        """
        Perform the back-tracking algorithms to find all possible solutions to
        the CSP.

        @param assignment: A list of current assignment. len(assignment) should
            equal to self.csp.numVars. Unassigned variables have None values, while an
            assigned variable has the index of the value with respect to its
            domain. e.g. if the domain of the first variable is [5,6], and 6
            was assigned to it, then assignment[0] == 1.
        @param numAssigned: Number of currently assigned variables
        @param weight: The weight of the current partial assignment.
        """

        self.numOperations += 1
        assert weight > 0
        if numAssigned == self.csp.numVars:
            # A satisfiable solution have been found. Update the statistics.
            self.numAssignments += 1
            newAssignment = {}
            for var in range(self.csp.numVars):
                newAssignment[self.csp.varNames[var]] = self.csp.valNames[var][assignment[var]]
            self.allAssignments.append(newAssignment)
            self.allWeights.append(weight)

            if len(self.optimalAssignment) == 0 or weight >= self.optimalWeight:
                if weight == self.optimalWeight:
                    self.numOptimalAssignments += 1
                else:
                    self.numOptimalAssignments = 1
                self.optimalWeight = weight

                self.optimalAssignment = newAssignment
                if self.firstAssignmentNumOperations == 0:
                    self.firstAssignmentNumOperations = self.numOperations
            return

        # Select the index of the next variable to be assigned.
        var = self.get_unassigned_variable(assignment)

        # Obtain the order of which a variable's values will be tried. Note that
        # this stores the indices of the values with respect to |var|'s domain.
        ordered_values = self.get_ordered_values(assignment, var)
        # Continue the backtracking recursion using |var| and |ordered_values|.
        if not self.mac:
            # When arc consistency check is not enabled.
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(assignment, var, val)
                if deltaWeight > 0:
                    assignment[var] = val
                    self.backtrack(assignment, numAssigned + 1, weight * deltaWeight)
                    assignment[var] = None
        else:
            # Problem 3.1d
            # When arc consistency check is enabled.
            # BEGIN_YOUR_CODE (around 10-15 lines of code expected)
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(assignment, var, val)
                # Check if this assignment of val to var is  consistent with current assignments
                if deltaWeight > 0:
                    assignment[var] = val
                    tempDomain = copy.deepcopy(self.domains)
                    self.domains[var] = [val]   # Since this var is assigned val, the domain becomes just [val]
                    self.arc_consistency_check(var)     # Update other variable's domain based on this newly assigned var
                    self.backtrack(assignment, numAssigned + 1, weight * deltaWeight)      # Recursively dfs backtrack
                    assignment[var] = None      # Restore assignment of var to None
                    self.domains = tempDomain   # Restore domains back before the assignment
            # raise Exception("Not implemented yet")
            # END_YOUR_CODE

    def get_unassigned_variable(self, assignment):
        """
        Given a partial assignment, return the index of a currently unassigned
        variable.

        @param assignment: A list of current assignment. This is the same as
            what you've seen so far.

        @return var: Index of a currently unassigned variable.
        """

        if not self.mcv:
            # Select a variable without any heuristics.
            for var in range(len(assignment)):
                if assignment[var] is None: return var
        else:
            # Problem 3.1b
            # Heuristic: most constrained variable (MCV)
            # Select a variable with the least number of remaining domain values.
            # BEGIN_YOUR_CODE (around 5 lines of code expected)
            least_v_cnt = 10000
            mcv_var = 10000
            for var in range(len(assignment)):
                v_cnt = 0
                if assignment[var] is None:
                    for val in self.csp.valNames[var]:
                        idx_of_val = val - 1
                        if self.get_delta_weight(assignment, var, idx_of_val) != 0:
                            v_cnt += 1
                    if v_cnt < least_v_cnt:
                        least_v_cnt = v_cnt
                        mcv_var = var

            return mcv_var

            # raise Exception("Not implemented yet")
            # END_YOUR_CODE

    def get_ordered_values(self, assignment, var):
        """
        Given an unassigned variable and a partial assignment, return an ordered
        list of indices of the variable's domain such that the backtracking
        algorithm will try |var|'s values according to this order.

        @param assignment: A list of current assignment. This is the same as
            what you've seen so far.
        @param var: The variable that's going to be assigned next.

        @return ordered_values: A list of indices of |var|'s domain values.
        """
        if not self.lcv:
            # Return an order of value indices without any heuristics.
            # if var == None:
            #     import pdb; pdb.set_trace()
            return self.domains[var]
        else:
            # Problem 3.1c
            # Heuristic: least constraining value (LCV)
            # Return value indices in ascending order of the number of additional
            # constraints imposed on unassigned neighboring variables.
            # BEGIN_YOUR_CODE (around 12 lines of code expected)
            domain_values = self.csp.valNames[var]

            # domain_list = []
            # For each value a in the domain of the variable X_j we want to assign:
                # Create counter cnt = 0
                # For each unassigned variable X_k
                    # For each value b in its domain
                        # If (X_k = b) is consistent with (X_j = a): cnt++
                # domain_list.append([a, cnt])
            # return domain_list sorted by cnt

            tmp_domain_store = []  # temporary store: (domain value a for X_j,
                                   # count of values of an unassigned variable that is consistent with new assignment and original partial assignment
            for val in self.domains[var]:   # For each value a in X_j's domain
                cnt = 0
                for var_k in range(len(assignment)):
                    if var_k != var and assignment[var_k] is None:  # Now var_k hasn't been assigned yet
                        for val_k in self.domains[var_k]:  # need to check if val_k is already consistent with partial assignment
                            # 1st condition: choosing val_k for var_k should be consistent with previous assignment
                            # 2nd condition: choosing val_k for var_k should be consistent with choosing val for var
                            if self.get_delta_weight(assignment, var_k, val_k) != 0 and self.csp.binaryPotentials[var][var_k][val][val_k] == 0:
                                cnt += 1

                tmp_domain_store.append((val, cnt))
            tmp_domain_store_sorted = sorted(tmp_domain_store, key=lambda val: val[1])   # sort the domain values based on cnt
            domain_sorted = [i[0] for i in tmp_domain_store_sorted]
            return domain_sorted    # Modified the domain(?)

            # Will update the domains! The unary constraint on var, val was already checked by backtrack before calling this method
            # raise Exception("Not implemented yet")
            # END_YOUR_CODE
   
    def arc_consistency_check(self, var):
        """
        Perform the AC-3 algorithm. The goal is to reduce the size of the
        domain values for the unassigned variables based on arc consistency.

        @param var: The variable whose value has just been set.

        While not required, you can also choose to add return values in this
        function if there's a need.
        '''
            csp.binaryPotentials = 
            [{1: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              5: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {0: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              2: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              5: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {1: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              3: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              5: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {2: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              4: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              5: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {3: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              5: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {0: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              1: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              2: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              3: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
              4: [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]},

             {}]
        '''
        """
        # Problem 3.1d
        # BEGIN_YOUR_CODE (around 15-20 lines of code expected)

        changedVars = []        # If neighbor's domain changed -> put into the list because we need to do arc consistency on that neighbor later
        for neighbor_var in self.csp.binaryPotentials[var]:     # Loop through those that has constraints with var
            neighbor_domain_changed = False
            neighbor_val_to_del = []
            for neighbor_val in self.domains[neighbor_var]:
                consistent_cnt = 0
                for val in self.domains[var]:
                    if self.csp.binaryPotentials[var][neighbor_var][val][neighbor_val] != 0:
                        consistent_cnt += 1
                if consistent_cnt == 0:  # we remove neighbor_val from domain of neighbor_var because not consistent
                    neighbor_domain_changed = True
                    neighbor_val_to_del.append(neighbor_val)
            for valBIndex in neighbor_val_to_del:
                self.domains[neighbor_var].remove(valBIndex)
            if neighbor_domain_changed:
                changedVars.append(neighbor_var)
        for changed in changedVars:
            self.arc_consistency_check(changed)
        return

        # END_YOUR_CODE

############################################################
# Problem 3.2

def get_sum_variable(csp, name, variables, maxSum):
    """
    Given a list of |variables| each with non-negative integer domains,
    returns the name of a new variable with domain [0, maxSum], such that
    it's consistent with the value |n| iff the assignments for |variables|
    sums to |n|.

    @param name: Prefix of all the variables that are going to be added.
        Can be any hashable objects. For every variable |var| added in this
        function, it's recommended to use a naming strategy such as
        ('sum', |name|, |var|) to avoid conflicts with other variable names.
    @param variables: A list of variables that are already in the CSP that
        have non-negative integer values as its domain.
    @param maxSum: An integer indicating the maximum sum value allowed.

    @return result: The name of a newly created variable with domain
        [0, maxSum] such that it's consistent with an assignment of |n|
        iff the assignment of |variables| sums to |n|.
    """

    # BEGIN_YOUR_CODE (around 12-15 lines of code expected)
    ## WE NEED TO FIX THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if len(variables) is 0:
        varName = 'sum' + str(name)
        csp.add_variable(varName, [0])
        return varName

    varName = 'sum' + str(name) + '0'
    csp.add_variable(varName, [(0, i) for i in range(maxSum + 1)])
    csp.add_binary_potential(varName, variables[0], lambda x, y: x[1] == y)

    for i, var in enumerate(variables):
        if i > 0:
            oldVarName = varName
            varName = 'sum' + str(name) + str(i)
            csp.add_variable(varName, [(k, j) for k in range(maxSum + 1) for j in range(maxSum + 1)])
            csp.add_binary_potential(varName, oldVarName, lambda x, y: x[0] == y[1])
            csp.add_binary_potential(varName, variables[i], lambda x, y: x[1] == (x[0] + y))

    lastVarName = 'sum' + str(name)
    csp.add_variable(lastVarName, range(maxSum + 1))
    csp.add_binary_potential(lastVarName, varName, lambda x, y: x == y[1])
    return lastVarName


    # raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3.3

# A class providing methods to generate CSP that can solve the course scheduling
# problem.
class SchedulingCSPConstructor():

    def __init__(self, bulletin, profile):
        """
        Saves the necessary data.

        @param bulletin: Rice Bulletin that provides a list of courses
        @param profile: A student's profile and requests
        """
        self.bulletin = bulletin
        self.profile = profile

    def add_variables(self, csp):
        """
        Adding the variables into the CSP. Each variable, (req, course),
        can take on the value of one of the semesters in req or None.
        For instance, for course='COMP310', and a request object, req, generated
        from 'in Fall2018,Fall2019', then (req, course) should have the domain values
        ['Fall2018', 'Fall2019', None]. Conceptually, if var is assigned 'Fall2018'
        then it means we are taking 'COMP310' in 'Fall2018'. If it's None, then
        we not taking COMP310.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for req in self.profile.requests:
            # print(req)   -->   Request{['COMP182'] [] [] 1}
            for cid in req.cids:
                if cid not in csp.varNames:
                    csp.add_variable(cid, self.profile.semesters + [None])
                for cid2 in req.prereqs:
                    if cid2 not in self.profile.taken:
                        if cid2 not in csp.varNames:
                            csp.add_variable(cid2, self.profile.semesters + [None])
                #the values could be one of the semesters or None

    def add_bulletin_constraints(self, csp):
        """
        Add the constraints that a course can only be taken if it's offered in
        that semester.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for cid in csp.varNames:
            csp.add_unary_potential(cid, \
                lambda semester: semester is None or self.bulletin.courses[cid].is_offered_in(semester) )

    def get_basic_csp(self):
        """
        Return a CSP that only enforces the basic constraints that a course can
        only be taken when it's offered and that a request can only be satisfied
        in at most one semester.

        @return csp: A CSP where basic variables and constraints are added.
        """
        csp = util.CSP()
        self.add_variables(csp)
        self.add_bulletin_constraints(csp)
        return csp

    def add_semester_constraints(self, csp):
        """
        If the profile explicitly wants a request to be satisfied in some given
        semesters, e.g. Fall2018, then add constraints to not allow that request to
        be satisfied in any other semester.

        @param csp: The CSP where the additional constraints will be added to.
        """
        # # BEGIN_YOUR_CODE (around 4 lines of code expected)
        for req in self.profile.requests:
        #     # -------------------------For logging--------------------------#
        #     # print(req)   -->    Request{['COMP310'] ['Fall2019'] [] 1}
        #     # print(req.semesters)    --> ['Fall2019']
        #     # -------------------------For logging--------------------------#
            if len(req.semesters) > 0:
                # print(self.profile.semesters) --> ['Fall2019']
                for semester in self.profile.semesters:
                    if semester not in req.semesters:
                       csp.add_unary_potential(req.cids[0], lambda cid: cid == None)   # request won't satisfy in any other semesters
        # raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def add_request_weights(self, csp):
        """
        Incorporate weights into the CSP. By default, a request has a weight
        value of 1 (already configured in Request). You should only use the
        weight when one of the requested course is in the solution. A
        unsatisfied request should also have a weight value of 1.

        @param csp: The CSP where the additional constraints will be added to.
        """
        # BEGIN_YOUR_CODE (around 3 lines of code expected) 
        #print(self.profile.semesters) --> ['Fall2019']
        for req in self.profile.requests:
            if len(req.semesters)>0:
                for semester in self.profile.semesters:
                        csp.add_unary_potential(req.cids[0], lambda cid: req.weight if cid != None else 1.0)   # request won't satisfy in any other semesters
        # raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def add_prereq_constraints(self, csp):
        """
        Adding constraints to enforce prerequisite. A course can have multiple
        prerequisites. You can assume that all courses in req.prereqs are
        being requested. Note that req.prereqs apply to every single course in 
        req.cids. You cannot take a course in a semester unless all of its 
        prerequisites have been taken before that semester. You should write your 
        own function that check the values (i.e. semesters) of the course you 
        request and its prerequisites and make sure that the values of prerequisites 
        are smaller (e.g. Spr2019 is smaller than Fall2020) than that of the course 
        you request if not None.

        @param csp: The CSP where the additional constraints will be added to.
        """
        # BEGIN_YOUR_CODE (around 20 lines of code expected)


        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def add_credit_constraints(self, csp):
        """
        Add constraint to the CSP to ensure that the total number of credits are
        within profile.minUnits/maxmaxUnits, inclusively. The allowed range for
        each course can be obtained from bulletin.courses[id].minUnits/maxUnits.
        For a request 'Request A', if you choose to take A, then you must use a credit
        number that's within the range of A. You should introduce any additional
        variables that you are needed. In order for our solution extractor to
        obtain the number of credits, for every course that you plan to take in
        the solution, you must have a variable named (courseId, semester) (e.g.
        ('COMP182', 'Fall2018') and it's assigned value is the number of credits.
        You should take advantage of get_sum_variable().

        Note 1:
        In here, you can not use 
        for req in requests: 
            for course in req:
        to loop over all the courses, because prereqs are not added in to the instances 
        of request object.

        Note 2:
        So you will have to loop over variables in the csp. But there
        are different types of variables: courses and auxiliary variables with
        which you handle sums (e.g. (courseId, semester) and those added in get_sum_variable). 
        Please check the types of the variables before you work with them.
        

        @param csp: The CSP where the additional constraints will be added to.
        """
        # BEGIN_YOUR_CODE (around 13-15 lines of code expected)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def add_all_additional_constraints(self, csp):
        """
        Add all additional constraints to the CSP.

        @param csp: The CSP where the additional constraints will be added to.
        """
        self.add_semester_constraints(csp)
        self.add_request_weights(csp)
        self.add_prereq_constraints(csp)
        self.add_credit_constraints(csp)
