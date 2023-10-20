# Numerical Methods for Structural Economic Models
# Spring 2022
# Zach Stangebye
# Lecture 5: Machine Learning Example

# We'll solve the Arellano (2008) model using the Gu and Stangebye (2023) method 
# for GPR-VFI

const max_iters = 600

# Model parameters
const r = .017
const β = 0.953

const σ_CRRA = 2.0

function sov_utility(c)
    u = 1.0
    if c <= 0.0
        u = -1.0e10
    else
        u = c^(1-σ_CRRA)/(1-σ_CRRA)
    end
    return u
end

const σy = .025 # Output shock volatility
const ρy = .945 # Output persistence
const σy_uncond = σy/sqrt(1-ρy^2)

# const ythresh = 0.969 # Default cost parameter

# Approximately generates threshold-cost specification of Arellano (2008)
const y_const = -.5 
const y_curv = .53 
ydef(y) = y - max(0.0,y_const*y + y_curv*y^2)

const π_re_enter = .282 # Re-entry probability

const D = 2 #dimensionality of the problem

# Bounds on y, b, q, V
const stds_out = 3.0

const yL = exp(-stds_out*σy_uncond)
const yH = exp(stds_out*σy_uncond)

const bLopt = 0
const bHopt = 0.3 

const bL = bLopt-1.0e-6
const bH = bHopt+1.0e-6

const qL = 0.0
const qH = 1/(1+r)

const VL = -50.0 #-100.0
const VH = -1.0e-6

const bN_opt = 50
const b_opt_grid = bLopt:(bHopt-bLopt)/(bN_opt-1):bHopt

# ML parameters
const N_inputs = 30*D #
# Scheidigger and Bilionis (2019) suggest 5*D or 10*D for N_inputs, but we
# find this to be insufficient given the significan non-linearities in these models
# However, the golden feature that the problem is linear in D remains
const N_validate = 10000

using Random
using StatsFuns
using LinearAlgebra
using Optim
using FastGaussQuadrature
using Plots
using LaTeXStrings
gr() # Sets the backend for the plotting interface (not used in solution)

# Take expectations with a Gauss-Chebyshev quadrature
# Given that there are significantly fewer grid points, it's important
# for this to be reasonably large for precise expectations
const quad_size = 15
const gc_points, gc_weights = gausschebyshev(quad_size)

# Minor adjustment to quadrature-based expectations to ensure pdf integrates to unity
const pdf_adj = normcdf(stds_out)-normcdf(-stds_out)

eyeN1 = zeros(N_inputs,N_inputs)
for i=1:N_inputs
    eyeN1[i,i] = 1.0
end
const eyeN = copy(eyeN1)


# We start by drawing inputs/gridpoints (fixed for all iterations) from typical sets 
const typ_thresh = 1.0e-6 
const b_thresh = .75*bHopt-bLopt 
const N_thresh = 10*D

gridPointsTemp = zeros(2,N_inputs)
gridPointsTempVec = zeros(1,N_inputs)
# We draw debt gridpoints (endogenous) from a uniform grid, adding a few more points 
# in high-debt areas even though the soveregn never goes there
for i=1:N_inputs
    gridPointsTempVec[i] = yL + (yH - yL)*(i-1.0)/(N_inputs-1.0)
end 
for i=1:N_thresh
    gridPointsTemp[2,i] = bLopt + (b_thresh-bLopt)*(i-1.0)/(N_inputs-1.0)
end 
for i=N_thresh+1:N_inputs
    gridPointsTemp[2,i] = b_thresh + (bHopt-b_thresh)*(i-N_thresh-1)/(N_inputs-N_thresh-1)
end 

Random.seed!(13) # Set random seed so get same answer each time

#  y is drawn to be unconditionally typical
const z_entropy =  .5*log(2.0*π*ℯ*σy_uncond^2.0)
not_typical = true
while (not_typical)
    z_draw_entropy = 0.0
    for i=1:N_inputs 

        z_draw = σy_uncond*randn()
        gridPointsTemp[1,i] = exp( z_draw )
        z_draw_entropy = z_draw_entropy - log(normpdf(z_draw/σy_uncond)/σy_uncond)/N_inputs

    end 

    if (  abs(z_draw_entropy - z_entropy) < typ_thresh ) 

        global not_typical = false 

    end 
end 

const gridPoints = copy(gridPointsTemp)
const scaledGridPoints = (gridPointsTemp .- [yL;bL] )./([yH - yL; bH - bL])

# Finally, a function to draw random points from the state space to check for 
# convergence 
function drawConvergenceInputs(num_draws)
    xx = zeros(2,num_draws)
    for i=1:num_draws 
        xx[:,i] = rand(2,1).*[yH-yL;bH-bL] .+ [yL;bL]
    end
    return xx
end

# Here are the bounds on the hyper-parameters
const kernSSLogLB = -2.0
const kernSSLogUB = 8.0
const kernLogLB = -2.0
const kernLogUB = 8.0

const init_kernCoeff = [1.0;1.0] # Ensure this guess gives finite log-likelihood

const sn_star = 1.0e-4 # Assumed 'measurement' noise -> sn_star^2 = 1.0e-8

# Here's the covariance function (square exponential, no ARD)
function k_SE(x,xprime,θvec)
    s = θvec[1]
    l = θvec[2]
    return exp( -.5*sum( ( (x .- xprime)/l ).^2 ) + 2*log(s) )
end

# This function creates the covariance matrix, which is used both in likelihood maximization and in 
# the derivation of the GPR coefficients 
function createK(hyper_params,num_dims)

    currK = zeros(N_inputs,N_inputs)

    for ik1=1:N_inputs 
        for ik2=1:ik1 

            if (num_dims == 1) 
                currK[ik1,ik2] = k_SE(scaledGridPoints[1,ik1],scaledGridPoints[1,ik2],hyper_params)
            else 
                currK[ik1,ik2] = k_SE(scaledGridPoints[:,ik1],scaledGridPoints[:,ik2],hyper_params)
            end 

            if (ik1 != ik2) 
                currK[ik2,ik1] = currK[ik1,ik2]
            end 

        end 
    end 

    return currK

end  

# This is the general likelihood function we optimize over in the GPR. It is split into two functions. 
function negLL(log_hyper_params,t_set,num_dims) 

    negLL1 = 0.0

    if ( (log_hyper_params[1] > kernSSLogLB) && (log_hyper_params[1] < kernSSLogUB) && (log_hyper_params[2] > kernLogLB) && (log_hyper_params[2] < kernLogUB) )

        hyper_params = 10.0.^log_hyper_params

        currK =  createK(hyper_params,num_dims)

        Ssigma = currK + sn_star^2.0*eyeN

        # This way uses the standard log(|Sigma|) approach
        temp_num = det(Ssigma)

        if (temp_num > 0.0) 
            negLL1 = log(temp_num) + sum(t_set'/Ssigma*t_set)
        else 
            negLL1 = 1.2e10
        end  

    else # If it violates the bounds, return a big number
        negLL1 = 1.0e10 

    end  

    return negLL1

end 

# This is the general function approximation
function GPR_approx(x,AA,kern_params,xx) 

    GPR_approx1 = 0.0

    for i=1:N_inputs
        GPR_approx1 = GPR_approx1 + AA[i]*k_SE(x,xx[:,i],kern_params)
    end 

    return GPR_approx1
end   

# These vectors will hold the original (logit-transformed) output values
tset_VDz = zeros(N_inputs)
tset_VRz = zeros(N_inputs)
tset_qz = zeros(N_inputs)
tset_Az = zeros(N_inputs)

# These vectors will hold the cleaned (logit-transformed) output values
tset_VD = zeros(N_inputs)
tset_VR = zeros(N_inputs)
tset_q = zeros(N_inputs)
tset_A = zeros(N_inputs)

# These scalars are used to clean the output values 
mean_VD = 500.0 # With the logit transform, this corresponds to an initial zero
mean_VR = 500.0 # With the logit transform, this corresponds to an initial zero
mean_q = -500.0 # With the logit transform, this corresponds to an initial zero
mean_A = -500.0 # With the logit transform, this corresponds to an initial zero

std_VD = 1.0 
std_VR = 1.0 
std_q = 1.0 
std_A = 1.0

# These store the kernel coefficients for the approximations (signal strength, scalelength)
VR_kernCoeff = ones(2)
VD_kernCoeff = ones(2)
q_kernCoeff = ones(2)
A_kernCoeff = ones(2)

# These store the GPR coefficients for the approximation 
VR_gprCoeff = zeros(N_inputs)
VD_gprCoeff = zeros(N_inputs)
q_gprCoeff = zeros(N_inputs)
A_gprCoeff = zeros(N_inputs)


# Now we can represent the equilibrium objects 
function vRepay(xIn) # Repayment value (Y_t, B_t)

    xTransformed = zeros(2) 
    xTransformed[1] = (xIn[1] - yL)/(yH - yL)
    xTransformed[2] = (xIn[2] - bL)/(bH - bL)

    return VL + (VH - VL)/(1.0 + exp(-std_VR*GPR_approx(xTransformed,VR_gprCoeff,VR_kernCoeff,scaledGridPoints)-mean_VR))

end 

# Now we can represent the equilibrium objects 
function vDefault(xIn) # Default value (Y_t)

    xTransformed = (xIn - yL)/(yH - yL)

    return VL + (VH - VL)/(1.0 + exp(-std_VD*GPR_approx(xTransformed,VD_gprCoeff,VD_kernCoeff,scaledGridPoints[1,:]')-mean_VD))

end 

function q(xIn) # Pricing function (Y_t, B_t+1)

    xTransformed = zeros(2) 
    xTransformed[1] = (xIn[1] - yL)/(yH - yL)
    xTransformed[2] = (xIn[2] - bL)/(bH - bL)

    return qL + (qH - qL)/(1.0 + exp(-std_q*GPR_approx(xTransformed,q_gprCoeff,q_kernCoeff,scaledGridPoints)-mean_q))

end 

function Apol(xIn) # Borrowing policy (Y_t, B_t)

    xTransformed = zeros(2) 
    xTransformed[1] = (xIn[1] - yL)/(yH - yL)
    xTransformed[2] = (xIn[2] - bL)/(bH - bL)

    return bL + (bH - bL)/(1.0 + exp(-std_A*GPR_approx(xTransformed,A_gprCoeff,A_kernCoeff,scaledGridPoints)-mean_A))

end 

defFun(x) = Float64(vDefault(x[1]) > vRepay(x))

# Now we define their associated likelihoods 
negLL_VR(log_hyper_params) = negLL(log_hyper_params,tset_VR,2)
negLL_VD(log_hyper_params) = negLL(log_hyper_params,tset_VD,1)
negLL_q(log_hyper_params) = negLL(log_hyper_params,tset_q,2)
negLL_A(log_hyper_params) = negLL(log_hyper_params,tset_A,2)

## Having dealt with all GP technicalities, we can now define model-specific objects 

# First, a function that delivers the pricing function given current sovereign behavior in the next period
function new_q(current_y,issued_b)

    zL1 = ρy*log(current_y) - stds_out*σy
    zH1 = ρy*log(current_y) + stds_out*σy

    z_gc_points =  zL1 .+ ( 1.0 .+ gc_points)./2.0 .*(zH1-zL1)
    yvec = copy(z_gc_points)

    for iy=1:quad_size 

        ytempObj = defFun([exp(z_gc_points[iy]);issued_b])

        yvec[iy] = gc_weights[iy]*ytempObj*sqrt(1.0-gc_points[iy]^2.0)*normpdf( (z_gc_points[iy] - ρy*log(current_y))/σy )/σy/pdf_adj

    end 
    new_def1 = (zH1-zL1)/2.0*sum(yvec)

    return (1.0-new_def1)/(1.0+r)

end

function new_VR(current_y,current_b, issued_b)

    zL1 = ρy*log(current_y) - stds_out*σy
    zH1 = ρy*log(current_y) + stds_out*σy

    z_gc_points =  zL1 .+ ( 1.0 .+ gc_points)./2.0 .*(zH1-zL1)
    yvec = copy(z_gc_points)

    for iy=1:quad_size 

        ytempObj = max( vRepay([exp(z_gc_points[iy]);issued_b]), vDefault(exp(z_gc_points[iy]) ) )

        yvec[iy] = gc_weights[iy]*ytempObj*sqrt(1.0-gc_points[iy]^2.0)*normpdf( (z_gc_points[iy] - ρy*log(current_y))/σy )/σy/pdf_adj

    end 
    CV = (zH1-zL1)/2.0*sum(yvec)

    flow_u = sov_utility( current_y - current_b + q([current_y;issued_b])*issued_b )

    return flow_u + β*CV

end

function new_VD(current_y)

    zL1 = ρy*log(current_y) - stds_out*σy
    zH1 = ρy*log(current_y) + stds_out*σy

    z_gc_points =  zL1 .+ ( 1.0 .+ gc_points)./2.0 .*(zH1-zL1)
    yvec = copy(z_gc_points)

    for iy=1:quad_size 

        ytempObj = π_re_enter*vRepay([exp(z_gc_points[iy]);0.0]) + (1.0-π_re_enter)*vDefault(exp(z_gc_points[iy]) )

        yvec[iy] = gc_weights[iy]*ytempObj*sqrt(1.0-gc_points[iy]^2.0)*normpdf( (z_gc_points[iy] - ρy*log(current_y))/σy )/σy/pdf_adj

    end 
    CV = (zH1-zL1)/2.0*sum(yvec)

    flow_u = sov_utility( ydef( current_y)  )

    return flow_u + β*CV

end

# Now we can begin the VFI operating as the limit of a finie-horizon game 

v_old_points = zeros(N_validate)
v_new_points = zeros(N_validate)
const conv_tol = 1.0e-6 
ddist = 1.0
iiters = 0 
sstart0 = time()
while ( (ddist > conv_tol) && (iiters < max_iters)) 

    global iiters += 1

    convergence_points = drawConvergenceInputs(N_validate)
    for i=1:N_validate 
        v_old_points[i] = vRepay(convergence_points[:,i])
    end

    # First, we update the pricing equation (from penultimate period on)
    for i=1:N_inputs 
        ytod = gridPoints[1,i]
        bissued = gridPoints[2,i]

        if ( iiters > 1) 
            tset_qz[i] = -log((qH-qL)/(min( qH-1.0e-6, max( new_q(ytod,bissued), qL + 1.0e-6)-qL) ) - 1.0)
        else 
            tset_qz[i] = -log((qH-qL)/(1.0e-6-qL) - 1.0)
        end  
    end
    # Now we `clean' the outputs before running the GPR 
    global mean_q = sum(tset_qz)/N_inputs
    global std_q = sqrt( sum((tset_qz .- mean_q).^2.0)/(N_inputs-1.0) )
    if (std_q < 1.0e-6) 
        global std_q = 1.0 
    end
    global tset_q = (tset_qz .- mean_q)./std_q

    global rresultsq = optimize(negLL_q,init_kernCoeff,NelderMead())

    global opt_paramsq = rresultsq.minimizer 
    global opt_LLq = -rresultsq.minimum

    global q_kernCoeff = 10.0.^opt_paramsq
    global K_q = createK(q_kernCoeff,2)
    global q_gprCoeff = (K_q .+ sn_star^2.0 .*eyeN)\tset_q


     # Now, we update value and policy functions
     for i=1:N_inputs 
        ytod = gridPoints[1,i]
        btod = gridPoints[2,i]

        vtemp = zeros(bN_opt)
        max_i = 1
        best_v = -1.0e10 
        for iopt=1:bN_opt
            vtemp[iopt] = new_VR(ytod,btod, b_opt_grid[iopt])

            if (vtemp[iopt] >= best_v) 
                best_v = vtemp[iopt]
                max_i = iopt 
            end 
        end

        bL_local = bLopt 
        bH_local = bHopt
        if (max_i == 1) 
            bL_local = bLopt
            bH_local = b_opt_grid[2]
        elseif (max_i == bN_opt)  
            bL_local = b_opt_grid[bN_opt-1]
            bH_local = b_opt_grid[bN_opt]
        else 
            bL_local = b_opt_grid[max_i-1]
            bH_local = b_opt_grid[max_i+1]
        end 

        v_obj(bprime) = -new_VR(ytod,btod,bprime)
        
        rresults = optimize(v_obj,bL_local,bH_local)
        vnew = -rresults.minimum 
        anew = rresults.minimizer

        tset_VRz[i] = -log((VH-VL)/(vnew-VL) - 1.0)
        tset_Az[i] = -log((bH-bL)/(anew-bL) - 1.0)

        vdnew = new_VD(ytod)

        tset_VDz[i] = -log((VH-VL)/(vdnew-VL) - 1.0)

    end
    # Now we `clean' the outputs before running the GPR 
    global mean_VR = sum(tset_VRz)/N_inputs
    global std_VR = sqrt( sum((tset_VRz .- mean_VR).^2.0)/(N_inputs-1.0) )
    if (std_VR < 1.0e-6) 
        global std_VR = 1.0 
    end
    global tset_VR = (tset_VRz .- mean_VR)./std_VR

    global mean_VD = sum(tset_VDz)/N_inputs
    global std_VD = sqrt( sum((tset_VDz .- mean_VD).^2.0)/(N_inputs-1.0) )
    if (std_VD < 1.0e-6) 
        global std_VD = 1.0 
    end
    global tset_VD = (tset_VDz .- mean_VD)./std_VD

    global mean_A = sum(tset_Az)/N_inputs
    global std_A = sqrt( sum((tset_Az .- mean_A).^2.0)/(N_inputs-1.0) )
    if (std_A < 1.0e-6) 
        global std_A = 1.0 
    end
    global tset_A = (tset_Az .- mean_A)./std_A

    # Now, we optimize all the GP likelihoods and get the new functions
    global rresultsVR = optimize(negLL_VR,init_kernCoeff,NelderMead())

    global opt_paramsVR = rresultsVR.minimizer 
    global opt_LLVR = -rresultsVR.minimum

    global VR_kernCoeff = 10.0.^opt_paramsVR
    global K_VR = createK(VR_kernCoeff,2)
    global VR_gprCoeff = (K_VR .+ sn_star^2.0 .*eyeN)\tset_VR


    global rresultsVD = optimize(negLL_VD,init_kernCoeff,NelderMead())

    global opt_paramsVD = rresultsVD.minimizer 
    global opt_LLVD = -rresultsVD.minimum

    global VD_kernCoeff = 10.0.^opt_paramsVD
    global K_VD = createK(VD_kernCoeff,1)
    global VD_gprCoeff = (K_VD .+ sn_star^2.0 .*eyeN)\tset_VD


    global rresultsA = optimize(negLL_A,init_kernCoeff,NelderMead())

    global opt_paramsA = rresultsA.minimizer 
    global opt_LLA = -rresultsA.minimum

    global A_kernCoeff = 10.0.^opt_paramsA
    global K_A = createK(A_kernCoeff,2)
    global A_gprCoeff = (K_A .+ sn_star^2.0 .*eyeN)\tset_A


    # Now update the value function and check for convergence
    for i=1:N_validate 
        v_new_points[i] = vRepay(convergence_points[:,i])
    end
    global ddist = maximum(abs.(v_new_points .- v_old_points))
    println([iiters ddist])
end

tot_time_mins_vfi = (time() - sstart0)/60.0

# cd("/Users/zstangeb/Dropbox/AAA_CoursePrep/numerical_methods_class/lectures/machine_learning/code")


# Plot the solved policy and pricing functions
bN_plot = 500
b_plot_grid = bLopt:(bHopt-bLopt)/(bN_plot-1):bHopt

q_plot = zeros(bN_plot)
VR_plot = zeros(bN_plot)
VD_plot = zeros(bN_plot)
A_plot = zeros(bN_plot)
for i=1:bN_plot 
    q_plot[i] = q([1.0;b_plot_grid[i]])
    VR_plot[i] = vRepay([1.0;b_plot_grid[i]])
    A_plot[i] = Apol([1.0;b_plot_grid[i]])
    VD_plot[i] = vDefault(1.0)
end

b_grid = bL:.001:bH
# b_grid = bL:.03:bH
q_ss(b) = q([1.0, b])
q_l(b) = q([0.9, b])
q_h(b) = q([1.1, b])
plot(b_grid,q_ss.(b_grid),label="SS Output",xlabel="Debt Issuance",ylabel="Price")
plot!(b_grid,q_l.(b_grid),label="Low Output")
plot!(b_grid,q_h.(b_grid),label="High Output")

savefig("pricing_functions_arellano_2008.pdf")



a_b_alone_h(b) = Apol([1.1, b])
a_b_alone_l(b) = Apol([0.9, b])
plot(b_grid,a_b_alone_h.(b_grid),label="High Output",ylabel="Debt Issuance",xlabel="Debt Stock")
plot!(b_grid,a_b_alone_l.(b_grid),label="Low Output")

savefig("policy_functions_arellano_2008.pdf")


y_grid = yL:.001:yH
btest = 0.2
V_y_alone(y) = vRepay([y; btest])
Vd_y_alone(y) = vDefault(y)

plot(y_grid,V_y_alone.(y_grid),xlabel="Output",label="Repayment Value")
plot!(y_grid,Vd_y_alone.(y_grid),label="Default Value")

savefig("Value_functions_y_arellano_2008.pdf")
