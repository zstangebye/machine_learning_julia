# machine_learning_julia
Solution code for Arellano (2008) using the GPR-VFI method of Gu and Stangebye (2022). The algorithm is a Gaussian-Process Dynamic Programming solution in the vein of Scheidigger and Bilionis (2019). Significant alterations are made to improve performance for non-linear sovereign default models. See Appendix A in Gu and Stangebye (2022) for the relevant changes.

The code is designed with a modular structure to facilitate its translation to other dynamic non-linear models. The solution uses the mode of Gaussian Processes to approximate endogenous functions, such as value, price, and policy functions. To adapt the code for other purposes, proceed as follows.

This example is not parallelized for clarity but it can easily be made so both in the evaluation of the training inputs and the optimization of the GP likelihood. See the solution to Gu and Stangebye (2022) for details. 

TO CREATE A NEW FUNCTION in this vein (let's call it function G), proceed as follows:

NEW VARIABLES
1) Set lower and upper bounds for the function G: GL, GH
2) Create a 2-D global vector housing the current kernel coefficients for function G: G_kernCoeff
3) Define a covariance matrix: K_G. This will be used to generate the GPR coefficients
4) Create a global vector with length equal to the number of grid points to house the coefficients for the GPR approximation: G_gprCoeff
5) Create a global vector with length equal to the number of grid points to house the (logit-transformed) values of X at the grid points: tset_Gz
6) Create a pair of scalars to house the sample mean and standard deviation of tset_Gz: G_mean, G_std. These are used to clean, i.e., de-mean and standardize the logit-transformed output values of G, such that the mean-zero prior has no significant effect in interpolation
7) Create a global vector with length equal to the number of grid points to house the "cleaned" values of tset_Gz: tset_G. This is what the GPR will operate on
8) Normalize global mean and standard deviation parameters to zero and one respectively. Set G_gprCoeff to initial conditions (bearing in mind the logit-transform)


NEW FUNCTIONS

1) Create your new function to be called G(). Follow the template for vRepay or vDefault, i.e., translate input variables into the unit hypercube on which the GPR operates and evaluate the GPR at the current (a) kernel parameters, (b) GPR coefficients, (c) mean and standard-deviation scalars. Note GPR_approx takes the dimension of the problem as well as the hypercube-translated gridpoints as arguments
2) Create the function/subroutine that delivers new values for G given other current equilibrium objects, e.g., Bellman equation optimization, prices based on expectations, etc. This will be called in the VFI to provide data to run the GPR on
3) Create a log-likelihood function tailored to your function G: negLL_G(). Follow the template for negLL_VR() or negLL_VD(). These eliminate extra arguments from the general negLL() function so as to be suitable to be passed to the Nelder-Mead minimization in the main VFI.


CHANGES IN VFI
1) Add the update step for your function G in the VFI loop wherever it belongs. Follow the examples in the code for either new_q or whatever is nearest your application.
2) Store logit-transformed values in tset_Gz for all input points
3) Clean tset_Gz by de-meaning/standardizing it using G_mean and G_std to create tset_G. Again, follow the template already there 
4) Optimize the likelihood (negLL_G) of tset_G over the log-10 kernel parameters. This is done in a loop over all starting points with finite likelihoods. Again, follow the template for some other function, e.g., negLL_q
5) Use the optimal kernel parameters (G_kernCoeff) to create the covariance matrix K_G
6) Use the matrix K_G to create the GPR coefficients (G_gprCoeff). Once this is done, the function G() will be automatically updated and may be called from anywhere in the code, especially in the next loop.


TO ADD DIMENSIONS/STATES

1) Increase nDims to the highest number of dimensions required by any function
2) In the "drawInputs" function, add new dimensions as required. Endogenous states are set to be a (scrambled) uniform grid and exogenous states are drawn randomly until a typical set is reached. Follow existing templates.
3) Make similar changes in the "drawConvergenceInputs" function.
4) Following the call to drawInputs, scale the drawn inputs (gridPoints) into the unit hypercube (scaledGridPoints). Use extant code is a template. The GPR always operates on these scaled grid points.


TIPS

1) Different kernel functions can improve performance, but Automatic-Relevance-Determination (ARD) kernels often do not fare well as the likelihood is often maximized by flattening an entire dimension/state. To change the kernel, just change the function of k_SE() (and all associated calls) to whichever alternative kernel is preferred
2) Some smoother problems work better with fewer grid points; others require more. It is a parameter worth experimenting with from application to application
3) Logit-transforming is not always necessary, e.g., for unbounded value functions. I recommend it above for the sake of consistency/universality but it rarely affects performance. It is almost always necessary for bounded functions, though.
