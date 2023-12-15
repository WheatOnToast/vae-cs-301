"""Train variational autoencoder on binary MNIST data."""

import numpy as np
import random
import time

import torch
import torch.utils
import torch.utils.data
from torch import nn

import data
import flow
import argparse
import pathlib


def add_args(parser):
    parser.add_argument("--latent_size", type=int, default=128)
    parser.add_argument("--variational", choices=["flow", "mean-field"])
    parser.add_argument("--flow_depth", type=int, default=2)
    parser.add_argument("--data_size", type=int, default=784)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=582838)
    parser.add_argument("--train_dir", type=pathlib.Path, default="/tmp")
    parser.add_argument("--data_dir", type=pathlib.Path, default="/tmp")

'''
Model
- Defines a VAE that produces parameters for the distribution
- Calculates the normal and bernoulli log probabilities
- Essentially holds the main components of a VAE
'''

class Model(nn.Module):
    """Variational autoencoder, parameterized by a generative network."""

    def __init__(self, latent_size, data_size):
        super().__init__()
        self.register_buffer("p_z_loc", torch.zeros(latent_size)) ## the mean of a distribution 
        self.register_buffer("p_z_scale", torch.ones(latent_size)) ## the standard deviation of a distribution
        self.log_p_z = NormalLogProb() ## normal log probability
        self.log_p_x = BernoulliLogProb() ## bernoulli log probability
        self.generative_network = NeuralNetwork( ## neural network instance
            input_size=latent_size, output_size=data_size, hidden_size=latent_size * 2
        )

    def forward(self, z, x):
        """Return log probability of model."""
        log_p_z = self.log_p_z(self.p_z_loc, self.p_z_scale, z).sum(-1, keepdim=True) ## normal log probability
        logits = self.generative_network(z)
        # unsqueeze sample dimension
        logits, x = torch.broadcast_tensors(logits, x.unsqueeze(1)) 
        log_p_x = self.log_p_x(logits, x).sum(-1, keepdim=True) ## bernoulli log probability
        return log_p_z + log_p_x # returns (bernoulli + normal) log probabilities


'''
VariationalMeanField
- creates an instance of the neural network class
- finds the variational parameters and minimize divergence
- this will return the sampled variables and log_q_z which is the probability density of the samples variables
'''

class VariationalMeanField(nn.Module):
    """Approximate posterior parameterized by an inference network."""

    def __init__(self, latent_size, data_size):
        super().__init__()
        self.inference_network = NeuralNetwork( # neural network instance
            input_size=data_size,
            output_size=latent_size * 2,
            hidden_size=latent_size * 2,
        )
        self.log_q_z = NormalLogProb() # normal log probability
        self.softplus = nn.Softplus() # makes sure output is positive

    def forward(self, x, n_samples=1):
        """Return sample of latent variable and log prob."""
        loc, scale_arg = torch.chunk(
            self.inference_network(x).unsqueeze(1), chunks=2, dim=-1
        )
        scale = self.softplus(scale_arg) #makes sure scale is positive
        eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]), device=loc.device)
        z = loc + scale * eps  # reparameterization
        log_q_z = self.log_q_z(loc, scale, z).sum(-1, keepdim=True)
        return z, log_q_z # returns normal log probability and z

'''
VariationalFlow
- implements inverse regressive flows
- Converts variables z_0 into a set of variables as z_T
'''

class VariationalFlow(nn.Module):
    """Approximate posterior parameterized by a flow (https://arxiv.org/abs/1606.04934)."""

    def __init__(self, latent_size, data_size, flow_depth):
        super().__init__()
        hidden_size = latent_size * 2
        self.inference_network = NeuralNetwork( #neural network instance
            input_size=data_size,
            # loc, scale, and context
            output_size=latent_size * 3,
            hidden_size=hidden_size,
        )
        modules = []
        for _ in range(flow_depth):
            modules.append(
                flow.InverseAutoregressiveFlow( #inverse autoregressive flow
                    num_input=latent_size,
                    num_hidden=hidden_size,
                    num_context=latent_size,
                )
            )
            modules.append(flow.Reverse(latent_size))
        self.q_z_flow = flow.FlowSequential(*modules) #sequence of inverse regressive flow layers
        self.log_q_z_0 = NormalLogProb() #normal log probability
        self.softplus = nn.Softplus() #ensures positivity

    def forward(self, x, n_samples=1): #finds the log probability of the transformed variables
        """Return sample of latent variable and log prob."""
        loc, scale_arg, h = torch.chunk(
            self.inference_network(x).unsqueeze(1), chunks=3, dim=-1
        )
        scale = self.softplus(scale_arg)
        eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]), device=loc.device)
        z_0 = loc + scale * eps  # reparameterization
        log_q_z_0 = self.log_q_z_0(loc, scale, z_0) #log probability of the variables z_0 based off parameters loc and scale
        z_T, log_q_z_flow = self.q_z_flow(z_0, context=h)
        log_q_z = (log_q_z_0 + log_q_z_flow).sum(-1, keepdim=True) #total log probability
        return z_T, log_q_z #returns the final set of transformed variables and the total log probability


'''
NeuralNetwork
- neural network class 
- has multiple layers and ReLU functions
'''

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size): #constructor that takes size of input feature, size of output, and size of hidden layers
        super().__init__()
        modules = [ #array of linear transformations of the parameters
            nn.Linear(input_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        ]
        self.net = nn.Sequential(*modules) #applies the linear transformations

    def forward(self, input): 
        return self.net(input)


class NormalLogProb(nn.Module): #calculates log probability of z under normal distribution with loc and scale
    def __init__(self):
        super().__init__()

    def forward(self, loc, scale, z):
        var = torch.pow(scale, 2)
        return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - loc, 2) / (2 * var) #returns log prob. density

'''
BernoulliLogProb
- the probability distribution of a single pixel 
- used in part with the decoder
- outputs around 784 bernoulli parameters
'''

class BernoulliLogProb(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="none") #binary cross entropy is used

    def forward(self, logits, target):
        # bernoulli log prob is equivalent to negative binary cross entropy
        return -self.bce_with_logits(logits, target) #returns BCE of logits and target (logits are log-odds values)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


'''
Evaluate
- evaluates the VAE model
- estimates ELBO and log prob. of a data set
'''
@torch.no_grad()
def evaluate(n_samples, model, variational, eval_data): #takes number of samples, an evaulated model, a model used for variational inference, and a data set
    model.eval() #evaluation mode
    total_log_p_x = 0.0
    total_elbo = 0.0
    for batch in eval_data:
        x = batch[0].to(next(model.parameters()).device) #batch of evaluation data moved to the same device as the model's parameters
        z, log_q_z = variational(x, n_samples) #gets z and their probabilities 
        log_p_x_and_z = model(z, x) #gets log prob. of x and z
        # importance sampling of approximate marginal likelihood with q(z)
        # as the proposal, and logsumexp in the sample dimension
        elbo = log_p_x_and_z - log_q_z
        log_p_x = torch.logsumexp(elbo, dim=1) - np.log(n_samples) #prob. of data sampled based off of ELBO
        # average over sample dimension, sum over minibatch
        total_elbo += elbo.cpu().numpy().mean(1).sum()
        # sum over minibatch
        total_log_p_x += log_p_x.cpu().numpy().sum()
    n_data = len(eval_data.dataset)
    return total_elbo / n_data, total_log_p_x / n_data #returns average ELBO and log prob. over the data set


'''
main method
'''

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    add_args(parser)
    cfg = parser.parse_args()

    device = torch.device("cuda:0" if cfg.use_gpu else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    model = Model(latent_size=cfg.latent_size, data_size=cfg.data_size) # VAE model creation

    #arguments needed for cfg
    if cfg.variational == "flow": 
        variational = VariationalFlow(
            latent_size=cfg.latent_size,
            data_size=cfg.data_size,
            flow_depth=cfg.flow_depth,
        )
    elif cfg.variational == "mean-field":
        variational = VariationalMeanField(
            latent_size=cfg.latent_size, data_size=cfg.data_size
        )
    else:
        raise ValueError(
            "Variational distribution not implemented: %s" % cfg.variational
        )

    #puts model and variational to a specificed pytorch device
    model.to(device)
    variational.to(device)

    # RMS optimization
    optimizer = torch.optim.RMSprop(
        list(model.parameters()) + list(variational.parameters()),
        lr=cfg.learning_rate,
        centered=True,
    )


    #retrieves MNIST data set if it isn't downloaded
    fname = cfg.data_dir / "binary_mnist.h5"
    if not fname.exists():
        print("Downloading binary MNIST data...")
        data.download_binary_mnist(fname) #download mnist data
    train_data, valid_data, test_data = data.load_binary_mnist( #loads MNIST data 
        fname, cfg.batch_size, cfg.test_batch_size, cfg.use_gpu
    )

    best_valid_elbo = -np.inf
    num_no_improvement = 0  #keeps track of iterations with no improvement
    train_ds = cycle(train_data) #uses the cycle function to make an infinite loop over the training data
    t0 = time.time() #start time of iterations

    for step in range(cfg.max_iterations):
        batch = next(train_ds)  #gets next batch of data from train_ds
        x = batch[0].to(device) #moves the data to the pytorch device
        model.zero_grad() #gradient clearing
        variational.zero_grad() #gradient clearing
        z, log_q_z = variational(x, n_samples=1) #log probabilities of samples 
        log_p_x_and_z = model(z, x) #joint log probability of x and z
        # average over sample dimension
        elbo = (log_p_x_and_z - log_q_z).mean(1)
        # sum over batch dimension
        loss = -elbo.sum(0)
        loss.backward()
        optimizer.step()

        if step % cfg.log_interval == 0: #if step is a multiple of the log interval (if theres a logging point)
            t1 = time.time() #records time
            examples_per_sec = cfg.log_interval * cfg.batch_size / (t1 - t0) #speed of examples
            with torch.no_grad():
                valid_elbo, valid_log_p_x = evaluate( #evaluates model's performance and obtains ELBO and log prob. of a data set
                    cfg.n_samples, model, variational, valid_data
                )
            print(
                f"Step {step:<10d}\t"
                f"Train ELBO estimate: {elbo.detach().cpu().numpy().mean():<5.3f}\t"
                f"Validation ELBO estimate: {valid_elbo:<5.3f}\t"
                f"Validation log p(x) estimate: {valid_log_p_x:<5.3f}\t"
                f"Speed: {examples_per_sec:<5.2e} examples/s"
            )
            if valid_elbo > best_valid_elbo: #if ELBO improves
                num_no_improvement = 0  #counter reset
                best_valid_elbo = valid_elbo 
                states = {
                    "model": model.state_dict(),
                    "variational": variational.state_dict(),
                }
                torch.save(states, cfg.train_dir / "best_state_dict")
            t0 = t1

    checkpoint = torch.load(cfg.train_dir / "best_state_dict") #loads best model
    model.load_state_dict(checkpoint["model"])
    variational.load_state_dict(checkpoint["variational"])
    test_elbo, test_log_p_x = evaluate(cfg.n_samples, model, variational, test_data) #evaluates model and distribution on the data set

    #prints iteration statistics
    print(
        f"Step {step:<10d}\t"
        f"Test ELBO estimate: {test_elbo:<5.3f}\t"
        f"Test log p(x) estimate: {test_log_p_x:<5.3f}\t"
    )

    print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")
