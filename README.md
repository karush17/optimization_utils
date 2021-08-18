## Optimization Utilities

This repository implements a set of optimization tools and utilities using PyTorch and Autograd frameworks. Implementation of tools is provided in a simple and easy-to-read manner for future development. Most algorithms implemented can be found in textbook on [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) by Stephen Boyd & Lieven Vandenberghe or [Numerical Optimization](http://www.apmath.spbu.ru/cnsa/pdf/monograf/Numerical_Optimization2006.pdf) by Jorge Nocedal & Stephen Wright.

## Organization

Following are the utilities provided in the current version-

### Line Search

|Utility|Description|Location|
|:-----:|:---------:|:------:|
|BacktrackLineSearch|Armijo's line search with termination criterion|[`backtrack_line_search`](line_search/backtrack_line_search.py)|
|DiffLineSearch|Modified Armijo's line search with differential steps|[`diff_line_search`](line_search/diff_line_search.py)|

### Vector Products

|Utility|Description|Location|
|:-----:|:---------:|:------:|
|vjp|Vector-Jacobian Product|[`vjp`](vec_prod/vjp.py)|
|jvp|Jacobian-Vector Product|[`jvp`](vec_prod/jvp.py)|
|hvp|Hessian-Vector Product|[`hvp`](vec_prod/hvp.py)|
|vhp|Vector-Hessian Product|[`vhp`](vec_prod/vhp.py)|

### Descent Methods

|Utility|Description|Location|
|:-----:|:---------:|:------:|
|GradDescent|Stochastic Gradient Descent|[`grad_descent`](methods/desc_methods/grad_descent.py)|
|LangAscent|Stochastic Gradient Langevin Dynamics|[`lang_ascent`](methods/desc_methods/lang_ascent.py)|
|NaiveGaussNewtDescent|Naive Gauss-Newton Descent|[`gn_descent`](methods/desc_methods/gn_descent.py)|
|NewtDescent|Newton-Raphson Descent with approximate Newton step|[`app_newt_descent`](methods/desc_methods/app_newt_descent.py)|
|cg_solve|Conjugate Gradient Solver|[`cg_solver`](methods/desc_methods/cg_solver.py)|

### Primal-Dual Methods

|Utility|Description|Location|
|:-----:|:---------:|:------:|
|PrimalDualConstrainedDescent|Simultaneous descent on primal/dual problems for constraint satisfaction|[`constraint_opt`](methods/prim_dual/constraint_opt.py)|
|TunedLagrangianDescent|Simultaneous descent on Lagrange multiplier for automatic tuning|[`lagrange_opt`](methods/prim_dual/lagrange_opt.py)|
|ConstrainedLagrangianDescent|Simultaneous descent on primal/dual problems and Lagrange multiplier Lagrange multiplier|[`constraint_lagrange_opt`](methods/prim_dual/constraint_lagrange_opt.py)|

## Usage

To run Newton's descent method with backtracking line search on a simple MNIST classification problem, use the following-
```
python3 main.py --algo 'NewtDescent' --search_algo 'BacktrackLineSearch'
```

The default configuration runs gradient descent with line search for MNIST classification.
```
python3 main.py --algo 'GradDescent' --search_algo 'BacktrackLineSearch'
```

## Reference

In case you find these implementations, then please cite the following-
```
@misc{karush17opt,
  author = {Karush Suri},
  title = {Optimization Utilities},
  year = {2021},
  howpublished = {\url{https://github.com/karush17/optimization_utils}}
}
```
