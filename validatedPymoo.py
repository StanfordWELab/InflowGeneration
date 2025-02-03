from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from pymoo.util.plotting import plot
import numpy as np

class MyProblem(Problem):
    
    def f1(self,x):
        return x[:, 0]*1.0
    
    def temp(self,x):
        return 1.0+9.0/(30.0-1)*np.sum(x[:,1:],axis=1)
    
    def f2(self,x):
        return self.temp(x)*(1-np.sqrt(self.f1(x)/self.temp(x)))

    def __init__(self, x_low, x_up):
        
        super().__init__(n_var=30,
                         n_obj=2,
                         xl=x_low,
                         xu=x_up)

    def _evaluate(self, x, out, *args, **kwargs):
        
        out["F"] = np.column_stack([self.f1(x), self.f2(x)])
        
algorithm = NSGA2(pop_size=100)

problemUser = MyProblem(np.array([0.0]*30),np.array([1.0]*30))
resUser = minimize(problemUser,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

problemRef = get_problem("zdt1")
resRef = minimize(problemRef,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

plot = Scatter()
plot.add(resRef.F, facecolor="none",edgecolor='tab:blue')
plot.add(resUser.F, color='tab:orange', marker = 'x')
plot.show()

## create the algorithm object
#algorithm = NSGA2(pop_size=100)

## let the algorithm object never terminate and let the loop control it
#termination = NoTermination()

## create an algorithm object that never terminates
#algorithm.setup(problem, termination=termination)

## fix the random seed manually
#np.random.seed(1)

## until the algorithm has no terminated
#for n_gen in range(10):
    ## ask the algorithm for the next solution to be evaluated
    #pop = algorithm.ask()

    ## get the design space values of the algorithm
    #X = pop.get("X")

    ## implement your evluation. here ZDT1
    #f1 = X[:, 0]
    #v = 1 + 9.0 / (problem.n_var - 1) * np.sum(X[:, 1:], axis=1)
    #f2 = v * (1 - np.power((f1 / v), 0.5))
    #F = np.column_stack([f1, f2])

    #static = StaticProblem(problem, F=F)
    #Evaluator().eval(static, pop)

    ## returned the evaluated individuals which have been evaluated or even modified
    #algorithm.tell(infills=pop)

    ## do same more things, printing, logging, storing or even modifying the algorithm object
    #print(algorithm.n_gen)

## obtain the result objective from the algorithm
#res = algorithm.result()

## calculate a hash to show that all executions end with the same result
#print("hash", res.F.sum())
#print(res)
