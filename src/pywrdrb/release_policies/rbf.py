import numpy as np
import matplotlib.pyplot as plt

from pywrdrb.release_policies.abstract_policy import AbstractPolicy

# ## TODO: REPLACE THESE IMPORTS
from pywrdrb.release_policies.config import policy_n_params, policy_param_bounds, drbc_conservation_releases
from pywrdrb.release_policies.config import n_rbfs, n_rbf_inputs

class RBF(AbstractPolicy):
    """
    Radial Basis Function (RBF) policy class for reservoir operation.
    
    Uses Gaussian RBFs to determine scaled releases based on 
    input variables (x).
    
    One RBF is used for each input variable.
    
    Policy parameters are defined as:
    - c_ij: center of the ith RBF for the jth input (mean)
    - r_ij: width of the ith RBF for the jth input (standard deviation)
    - w_i: weight of the ith RBF (contribution to the output)
    
    All parameters are defined in the range [0, 1].
    
    The RBF function is defined as:
    z = sum(
        w_i * (
            sum(
                exp(-((x_j - c_ij) / r_ij)^2)
            ) for j in range(n_inputs)
        )
    ) for i in range(n_RBFs)
    
    
    Final release is computed as (subject to constraints):
    release = z * max_release
    
    More info on the formulation (for a different problem)
    can be found in Hadjimichael, Reed and Quinn (2020) 
    https://doi-org.proxy.library.cornell.edu/10.1155/2020/4170453
    """
    
    def __init__(self,
                 release_max,
                 release_min,
                 storage_capacity,
                 n_rbfs,
                 n_rbf_inputs,
                 policy_n_params,
                 policy_param_bounds,
                 policy_params):
        
        # Policy parameters
        self.nRBFs = n_rbfs
        self.n_params = policy_n_params["RBF"]
        self.param_bounds = policy_param_bounds["RBF"]
        self.n_inputs = n_rbf_inputs
        
        # X (input) max and min values
        # used to normalize the input data
        # X = [storage, inflow, day_of_year]
        # self.x_min = np.array([0.0, 
        #                     #    self.Reservoir.inflow_min,
        #                        1.0])
        
        # self.x_max = np.array([self.Reservoir.capacity, 
        #                        self.Reservoir.inflow_max,
        #                        366.0])
        
        self.policy_params = policy_params
        self.parse_policy_params()
        
    def validate_policy_params(self):
        """
        Validates the policy parameters
        """
        
        # Check if the number of parameters is correct
        assert len(self.policy_params) == self.n_params, \
            f"RBF policy expected {self.n_params} parameters, got {len(self.policy_params)}."
        
        # check parameter bounds
        for i, p in enumerate(self.policy_params):
            bounds = self.param_bounds[i]
            assert (p >= bounds[0]) and (p <= bounds[1]), \
                f"Parameter with index {i} is out of bounds {bounds}. Value: {p}."
        
        return
    
    def parse_policy_params(self):
        """
        Parses the policy parameters into RBF centers, widths, and weights.
        """
        
        # Validate the policy parameters
        self.validate_policy_params()
 
        ### Parse and assign
        # Given:
        # n RBF functions
        # d inputs (storage, inflow)
        # params = [ [w]*n, [c_ij]*n*d, [r_ij]*n*d ]
        
        w = self.policy_params[:self.nRBFs]
        
        start_idx = self.nRBFs
        end_idx = start_idx + (self.nRBFs)*self.n_inputs
        self.c = self.policy_params[start_idx:end_idx]
        
        start_idx = end_idx
        end_idx = start_idx + (self.nRBFs)*self.n_inputs
        self.r = self.policy_params[start_idx:end_idx]
        
        
        assert len(self.c) == self.nRBFs * self.n_inputs, \
            f"Expected {self.nRBFs * self.n_inputs} center parameters, got {len(self.c)}."
        assert len(self.r) == self.nRBFs * self.n_inputs, \
            f"Expected {self.nRBFs * self.n_inputs} radius parameters, got {len(self.r)}."
        assert len(w) == self.nRBFs, \
            f"Expected {self.nRBFs} weight parameters, got {len(w)}."
        
        
        # Make sure r > 0 to avoid division by zero
        for i in range(len(self.r)):
            self.r[i] = max(self.r[i], 1e-6)
        
        
        # Normalize the weights
        w_norm = []
        if np.sum(w) != 0:
            for w_i in w:
                w_norm.append(w_i / np.sum(w))
        else:
            w_norm = (1/self.nRBFs)*np.ones(len(w))
        self.w = w_norm
        
        return 

    def evaluate(self, X):
        """
        Evaluate the policy function.

        Args:
            X (list): A list of input values, including normalized:
                - Storage (S)
                - Inflow (I)
                - Day of year (D)

        Returns:
            float: The computed release.
        """
        
        # Make sure we got the right number of inputs
        assert len(X) == self.n_inputs, \
            f"Expected {self.n_inputs} input variables; got {len(X)}."

        # make sure X values are in [0, 1]
        assert all(0 <= x <= 1 for x in X), \
            f"Input values must be in the range [0, 1]. Values: {X}."
        
        # Calculate 
        z = 0.0
        for i in range(self.nRBFs):
            sq_term = 0.0
            for j in range(self.n_inputs):
                idx = i * self.n_inputs + j
                sq_term += ((X[j] - self.c[idx]) / self.r[idx]) ** 2
            z += self.w[i] * np.exp(-sq_term)
        
        # Impose bound limits
        z = max(0.0, min(1.0, z)) 
                       
        return z
    
    def get_release(self, 
                    inflow, 
                    storage,
                    day_of_year):
        
        # Get state variables
        I_t = float(inflow)
        S_t = float(storage)
        day_of_year = float(day_of_year)

        # inputs  = [storage, inflow, day_of_year]
        X = np.array([S_t, I_t, day_of_year])
        
        # Normalize X
        #TODO: Check the normalization logic 
        X_norm = np.zeros(self.n_inputs)
        for i in range(self.n_inputs):
            X_norm[i] = (X[i] - self.x_min[i]) / (self.x_max[i] - self.x_min[i])        
            X_norm[i] = max(0.0, min(1.0, X_norm[i])) # enforce bounds [0, 1]
        
        # Compute release
        release  = self.evaluate(X_norm) * self.release_max
        
        # Enforce constraints (defined in AbstractPolicy)
        release = self.enforce_constraints(release)
        release = min(release, S_t + I_t)
        release = max(release, self.release_min)

        return release

    def plot(self):
        
        # plot 3d surface
        
        fig, axs = plt.subplots(1, 1, 
                                figsize=(5, 5), 
                                subplot_kw={"projection": "3d"})
        
        xs = np.linspace(0.0, 1.0, 50)
        ys = np.linspace(0.0, 1.0, 50)
        zs = np.zeros((len(xs), len(ys)))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                X = np.array([x, y])
                zs[i,j] = self.evaluate(X)
        
        # plot surface
        X, Y = np.meshgrid(xs, ys)
        axs.plot_surface(X, Y, zs.T, cmap='viridis', alpha=0.5)
        
        axs.set_xlabel('Storage (normalized)')
        axs.set_ylabel('Inflow (normalized)')
        axs.set_zlabel('Release (normalized)')
        axs.set_title('RBF Policy Function')
        
        plt.tight_layout()
        plt.show()