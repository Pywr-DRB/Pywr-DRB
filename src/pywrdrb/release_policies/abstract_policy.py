from abc import ABC, abstractmethod

class AbstractPolicy(ABC):
    """
    Abstract policy class to enforce shared implementation 
    across different parameterized policies.
    """
    @abstractmethod
    def __init__(self, 
                 min_release,
                 max_release,
                 storage_capacity,
                 input_scaling_dict,
                 policy_params):
        """
        Initialize policy class.
        
        Parameters
        ----------
        policy_params : dict
            Dictionary of policy parameters.
        """
    
        self.release_min = None
        self.release_max = None
        self.storage_capacity = None
        self.input_scaling_dict = None
        self.x_min = None  # np.array([S_min, I_min, D_min])
        self.x_max = None  # np.array([S_max, I_max, D_max])

        self.release_min = float(release_min)
        self.release_max = float(release_max)
        self.storage_capacity = float(storage_capacity)
        self.input_scaling_dict = input_scaling_dict
        self.x_min = np.array(x_min, dtype=float)
        self.x_max = np.array(x_max, dtype=float)

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