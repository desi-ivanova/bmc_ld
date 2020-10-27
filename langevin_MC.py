import numpy as np
import torch
import torch.distributions as dist

def Langevin_MC(
    potential, 
    transform_fun=lambda x: x, 
    step=lambda x: 1e-2, 
    riemannian_metric=None,
    D=2, 
    n_samples=10000, 
    burnin_percent=0.1, 
    thining_percent=1, 
    x_0=None, 
    skip_MH=False,
    skip_noise=False,
):
    """
    Generic function to do (almost) all sorts of Langevin MC:
    - ULA : set skip_MH=True
    - MALA
    - Stochastic gradients: define step size
    - Transformed (e.g. mirror trick etc): specify transformation_fun (eg x->abs(x))
    PARAMS:
        potential: callable
        transform fun: callable
        step: callable
        
    RETURNS:
        np.array of samples
    """
    rejected = 0 # bookeeping
    burn_in = int(n_samples*burnin_percent)
    print_id = int(n_samples*.1)
    riemannian_metric = riemannian_metric or torch.ones(D, D)
    
    # initial proposal: sample random normal (start)
    if x_0 is None:
        x_0 = transform_fun(torch.randn(1, D))
    print(f"x_0={x_0}")
    x_next = x_0
    samples = torch.zeros(n_samples + burn_in, D)
    
    #noise_scale = np.sqrt(2 * step) ## might reverse..
    grad_step = step(0)
    def helper_q(x_prime, x, grad=None):
        """ Accept/reject """
        # for ref https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm
        # to see where it comes from notice that the proposal distribution is that noise term
        if grad is None:
            x.requires_grad_()
            grad = torch.autograd.grad(potential(x), x)[0]
        else:
            -(torch.norm(x_prime - x - grad_step*grad, p=2, dim=1)**2) / (4 * grad_step)
        #print(f"grad from helper {grad}")
        return -(torch.norm(x_prime - x - grad_step*grad, p=2, dim=1)**2) / (4 * grad_step)
    
    for i in range(n_samples + burn_in):
        grad_step = step(i)
        noise_scale = np.sqrt(2 * grad_step)
        x_next.requires_grad_()
        u = potential(x_next)
        
        grad = torch.autograd.grad(u, x_next)[0]
        #print(f"grad outside helper {grad}")
        if skip_noise:
            proposal = transform_fun(x_next.detach() + grad_step*grad)
        else:
            #                       |------- Gradient Update --------|+|-------- Noise --------------|    
            proposal = transform_fun(x_next.detach() + grad_step*grad + noise_scale*torch.randn(1, D))
        
        if skip_MH:
            x_next = proposal
        else:
            # accept reject ### 
            alpha = potential(proposal) - u + helper_q(x_next, proposal) - helper_q(proposal, x_next, grad)

            if torch.rand(1) < torch.exp(alpha):
                x_next = proposal
            else:
                rejected += 1
        
        samples[i] = x_next.detach()
        if i % print_id == 0:
            print(f"{i}/{n_samples} ... step size = {np.round(grad_step, 5)}; log posterior = {u.detach().numpy()}")
        
    print(f"rejections={np.round(100 * rejected / (n_samples + burn_in), 2)}%")
    return samples[burn_in:][::int(1/thining_percent)].numpy()
    
    
def step_function(t, gamma=1, a=1, b=1):
    """
    t: starts at 0
    gamma: has to be between (.5; 1]
    b: make sure denominator is not 0;
    """
    return a*(b + t)**(-gamma)

