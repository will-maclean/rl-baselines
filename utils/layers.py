import torch
from torch import nn


class NoisyLinear(nn.Module):

    def __init__(self, input_size, output_size, sigma_0=0.5):
        """
        Implementation of a noisy linear layer as described here: https://arxiv.org/pdf/1706.10295v3.pdf.

        Parameters
        ----------
        input_size: input size for linear layer
        output_size: output size for linear layer
        sigma_0: constant for bias initialisation

        Returns
        -------
        None
        """
        super(NoisyLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sigma_0 = sigma_0

        # if we want to be able to register variables that we want to be updated by optimisers, we wrap them in
        # nn.Parameter()
        self.weights_means = nn.Parameter(torch.zeros((self.output_size, self.input_size), dtype=torch.float))
        self.weights_sds = nn.Parameter(torch.zeros((self.output_size, self.input_size), dtype=torch.float))
        self.biases_means = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float))
        self.biases_sds = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float))

        # for variables that we want to keep track of, but not update, we use register_buffer(). These variables
        # are not present in model.params() (which a loss function uses) but are used in model.state_dict() (which
        # we use when copying)
        self.register_buffer('weights_epsilon', torch.zeros((self.output_size, self.input_size)))
        self.register_buffer('biases_epsilon', torch.zeros(self.output_size))

        self._init_params()
        self.update_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.weights_means
        biases = self.biases_means

        if self.training:
            # if we're training, add the random noise. If not, just use the expected value for the weights and biases
            weights = weights + self.weights_sds.mul(self.weights_epsilon)
            biases = biases + self.biases_sds.mul(self.biases_epsilon)

        return nn.functional.linear(x, weights, biases)  # computes x * transpose(weights) + biases

    def _init_params(self):
        """
        Initialises weights and biases parameters

        Returns
        -------
        None
        """
        dist_constant = 1/(self.input_size ** 0.5)  # constant used as distribution bounds when initialising means

        self.weights_means.data.uniform_(-dist_constant, dist_constant)  # uniform_ does what you'd expect
        self.biases_means.data.uniform_(-dist_constant, dist_constant)

        self.weights_sds.data.fill_(self.sigma_0)  # fill_ sets every value in the tensor to the scalar input
        self.biases_sds.data.fill_(self.sigma_0)

    def update_noise(self):
        """
        Updates the random noise being added to the weights and biases

        Returns
        -------
        None
        """
        # make two vectors of standard normal distribution, of size input and output. Scale this noise with the
        # _noise_func (see below). Couldn't tell you why we need to scale the noise, but the paper says to so here
        # we are
        input_noise = self._noise_func(torch.randn(self.input_size))
        output_noise = self._noise_func(torch.randn(self.output_size))

        # set the biases noise to the output noise vector created above
        self.biases_epsilon.copy_(output_noise)
        # set the weights noise to be the outer product of the output and input noise.
        # https://en.wikipedia.org/wiki/Outer_product
        self.weights_epsilon.copy_(output_noise.ger(input_noise))

    @staticmethod
    def _noise_func(x: torch.Tensor) -> torch.Tensor:
        """
        Scales the random noise. Using sign(x) * sqrt(abs(x)) as described in the paper
        Parameters
        ----------
        x: tensor of random noise

        Returns
        -------
        sign(x) * sqrt(abs(x))

        """
        return x.sign().mul(x.abs().sqrt())
