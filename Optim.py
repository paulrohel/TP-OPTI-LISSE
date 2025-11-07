import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def deriv_num(J,a,d,compute_grad=True,compute_Hess=True) :
    """test numerically the derivative and the Hessian of a function.
        
    Parameters
    ----------
    J : instance of a class
        The function to be tested it must have the following methods, where x is a 1d vector
        of size n
            -- J.eval(x) : evaluation of J at point x, must return a float
            -- J.grad(x) : evaluation of the gradient of J at point x, must a 1d vector of size n
            -- J.Hess(x) : evaluation of the Hessian of J at point x, typically a n*n matrix
    a : 1d vector of size n
        Point at which the numerical derivatives are evaluated
    d : 1d vector of size n
        Direction in which the numerical derivatives are evaluated
    compute_grad : Boolean
        Flag that tests the function J.grad against numerical derivatives
    compute_Hess : Boolean
        Flag that tests the function J.Hess against numerical derivatives of J.grad
    
   Ouput 
   -----
   This function does not have an output, it prints a string s.
    """
        
    eps_range=[0.1**(i+1) for i in range(12)]
    for eps in  eps_range:
        s='eps {:1.1e}'.format(eps)
        if compute_grad :
            # ratio of numerical derivatives of J and prediction given by J.grad
            ratio=(J.value(a+eps*d)-J.value(a))/(eps*np.dot(J.grad(a),d)) 
            s+=' grad {:1.1e}'.format(np.abs(ratio-1)) 
        if compute_Hess :
            v1=(J.grad(a+eps*d)-J.grad(a))/eps # numerical derivative of J.grad
            v2=J.Hess(a).dot(d)  # prediction given by J.Hess
            ratio=np.linalg.norm(v1)/np.linalg.norm(v2) #norm ratio
            angle=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)) # cosinus of the angle between the vectors
            s+=' ratio {:1.1e}'.format(np.abs(ratio-1.))
            s+=' angle {:1.1e}'.format(np.abs(angle-1.))
        print(s)
        
def main_algorithm(function,step,xini,dc,ls,itermax = 20000,tol=1.e-4,verbose=True):
    """Perform a minimization algorithm of a function
    
    Parameters
    ----------
    function : instance of a class
        The function to be minimized, depending on the choice of linesearch and direction of descent,
        it must have the following methods, where x is a 1d vector of size n
            -- function.eval(x) : evaluation of J at point x, must return a float
            -- function.grad(x) : evaluation of the gradient of J at point x, must a 1d vector of size n
            -- function.Hess(x) : evaluation of the Hessian of J at point x, typically a n*n matrix
    step : positive float
        Initial guess of the step
    xini : 1d vector of size n
        initial starting point
    dc : callable
        descent=dc(x,function,df,res)
        computes the descent direction with parameters 
           -x: the point x
           -df : the gradient of function at point x
           -function : the function
        The function dc returns
            -descent : the direction of descent 
    ls : callable
        x2,f2,df2,step2=ls(x, function, step, descent,f,df)
        performs a line search, the parameters are
           -x : initial point
           -step : initial step
           -function : the function to be minimized
           -f,df : the values of function(x) and the gradient of the function of x
           -descent : the descent direction
        the function returns
            -x2 : the new point x+step2*descent
            -f2 : the value of function at point x2
            -df2 : the value of the gradient of the function at point x2
            -step2 : the step given by the linesearch
    itermax : int
        maximum number of iterations
    tol : float
       stopping criterion
    verbose : Boolean
        Printing option of the algorithm
    
    Returns
    --------
    The function returns a single dictionnary res, the entries are
    res['list_x'] : list of 1d vectors of size n which are the iterates points of the algorithm
    res['list_steps'] : list of positive floats which are the different steps
    res['list_grads'] : list of positive floats which are the value of the euclidean norm of the gradients of the function
    res['final_x'] : 1d vector, final value of x      
    """
    x = xini
    res={'list_x':[],'list_steps':[],'list_costs':[],'list_grads':[],'final_x':[],'dc':[],'ls':[]}
    nbiter = 0
    f=function.value(x)
    df= function.grad(x)
    err=np.linalg.norm(df)
    if verbose :  print('iter={:4d} f={:1.3e} df={:1.3e} comp=[{:4d},{:4d},{:4d}]'.format(nbiter,f,err,function.nb_eval,function.nb_grad,function.nb_hess))
    res['list_x'].append(x.copy())
    res['list_costs'].append(f)
    res['list_grads'].append(err)
    while (err > tol) and (nbiter < itermax):
        descent = dc(x, function,df)
        x,f,df,step = ls(x, function, step, descent,f,df)
        err = np.linalg.norm(df)
        res['list_x'].append(x.copy())
        res['list_costs'].append(f)
        res['list_grads'].append(err)
        res['list_steps'].append(step)
        nbiter+=1
        if verbose : print('iter={:4d} f={:1.3e} df={:1.3e} comp=[{:4d},{:4d},{:4d}]'.format(nbiter,f,err,function.nb_eval,function.nb_grad,function.nb_hess))
        if (err <= tol):
            res['final_x']=np.copy(x)
            if verbose : print("Success !!! Algorithm converged !!!")
            return res
    if verbose : print("FAILED to converge")
    return res        

def graphical_info(res,function, levels=None,xmin=-2,xmax=2,ymin=-2,ymax=2):
    xiter=np.array(res['list_x'])
    fig, axarr = plt.subplots(2, 2, figsize=(16,8))
    # First plot 
    axarr[0,0].set_title('Points and levelset')
    Nx = 1000
    Ny = 1000
    x = np.linspace(xmin,xmax,Nx)
    y = np.linspace(ymin,ymax,Ny)
    X, Y = np.meshgrid(x, y)
    Z=function.value((X,Y))
    if levels:
        CS = axarr[0,0].contour(X, Y, Z, levels)
    else:
        CS = axarr[0,0].contour(X, Y, Z)
    axarr[0,0].plot(xiter[:,0], xiter[:,1],'+')
    axarr[0,0].clabel(CS, inline=1, fontsize=10)
    axarr[0,0].axis('equal')
    # Second plot
    axarr[0,1].set_title('Evolution of the cost')
    fiter=np.array(res['list_costs'])
    if min(fiter) > 0:
        axarr[0,1].semilogy(fiter)
    else:
        axarr[0,1].plot(fiter)
    #Third plot
    axarr[1,0].set_title('Norm of the gradient')
    giter=np.array(res['list_grads'])
    axarr[1,0].semilogy(giter)
    #Fourth plot
    axarr[1,1].set_title('Steps')
    siter=np.array(res['list_steps'])
    axarr[1,1].semilogy(siter)
    plt.show()