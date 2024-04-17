class CSP:
    def __init__(self):
        self.variables = {}  # name:domain

    def init_with_solutions(self, sols):
        nb_vars = len(sols[0])
        for i in range(nb_vars):
            self.variables[i] = [0,1]
        self.partial_solutions = sols.copy()
        self.partial_assignment = PartialAssigment()

    def check_mc_of(self, var, value):
        #count the number of solutions that satisfy both the partial assignment and the new variable value pair
        c = 0
        for s in self.partial_solutions:
            if s[var] == value:
                c+=1
        return c

    def extend_assignment(self, var, value, score):
        if var in self.partial_assignment.assigned:
            print("error")
            exit(-1)
        self.partial_assignment.assigned[var] = value
        temp = []
        for i,s in enumerate(self.partial_solutions):
            if s[var] == value:
                temp.append(s)
        self.partial_solutions = temp
        self.partial_assignment.score = score

    def extend_assignment_no_remove(self, var, value, score):
        if var in self.partial_assignment.assigned:
            print("error")
            exit(-1)
        self.partial_assignment.assigned[var] = value
        self.partial_assignment.score = score



#WE need a class for partial assignment
class PartialAssigment:
    def __init__(self):
        self.score = 0
        self.assigned = {} # dict (variable, value)




