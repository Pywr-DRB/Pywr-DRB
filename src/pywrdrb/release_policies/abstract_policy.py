from abc import ABC, abstractmethod

class AbstractPolicy(ABC):
    """
    Abstract policy class to enforce shared implementation 
    across different parameterized policies.
    """
    @abstractmethod
    def __init__(self, 
                 release_min,
                 release_max,
                 storage_capacity,
                 input_scaling_dict,
                 policy_n_params,
                 policy_param_bounds,
                 policy_params):
        """
        Initialize policy class.
        
        Parameters
        ----------
        release_min : float
            Minimum allowable release (e.g., conservation release).
        release_max : float
            Maximum allowable release (e.g., based on historical data).
        storage_capacity : float
            Maximum storage capacity of the reservoir.
        input_scaling_dict : dict
            Dictionary with min/max values for each input variable,
            used for normalization.
            Example: {"storage": (S_min, S_max), "inflow": (I_min, I_max), "day_of_year": (D_min, D_max)}
        policy_n_params : dict
            Dictionary with number of parameters for each policy type.
            Example: {"RBF": 7, "STARFIT": 17, "PiecewiseLinear": 25}
        policy_param_bounds : dict
            Dictionary with parameter bounds for each policy type.
            Example: {"RBF": [[0,1], [0,1], ..., [0,1]], "STARFIT": [[0,1], ..., [0,pi/2]], "PiecewiseLinear": [[0,1], ..., [0,pi/2]]}
        policy_params : list or np.array
            List or array of policy parameters to be parsed and used by the policy.
        """

        pass

    
    @abstractmethod
    def validate_policy_params(self):
        """
        Validate policy parameters, ensuring:
        - All required parameters are present
        - Parameters are of the correct type
        - Parameters are within valid ranges
        """
        pass
    
    @abstractmethod
    def parse_policy_params(self):
        """
        Parse policy parameters which will be provided 
        as an array of values. 

        Assign these values to the corresponding
        attribute variables.
        """
        pass


    def enforce_constraints(self, release):
        """
        Enforce constraints on the release.

        Args:
            release (float): The computed release.

        Returns:
            float: The release after enforcing max/min constraints.
        """
        return max(self.release_min, min(self.release_max, release))


    @abstractmethod
    def get_release(self, timestep):
        """
        Get the release for the current timestep,
        based on state information from the Reservoir object.
        
        Uses the evaluate method to compute the release,
        then enforces constraints on the release.
        """
        pass
    
    @abstractmethod
    def plot(self):
        """
        Plot the policy function f(state) -> release.
        """
        pass