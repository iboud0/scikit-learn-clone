import numpy as np

class ParameterGrid:
    """
    Grid of parameters with a discrete number of values for each.
    
    Parameters:
        param_grid : dict
            Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.
    
    Attributes:
        param_grid : dict
            The input parameter grid.
        keys : list
            The parameter names.
        param_grid_values : list
            The parameter settings to try for each parameter.
        grid_size : int
            The total number of parameter combinations in the grid.
    """
    def __init__(self, param_grid):
        self.param_grid = param_grid
        self.keys = list(param_grid)
        self.param_grid_values = [param_grid[key] for key in self.keys]
        self.grid_size = np.prod([len(v) for v in self.param_grid_values])

    def __iter__(self):
        """
        Generator that yields the next set of parameters in the grid.
        
        Yields:
            params : dict
                Dictionary where keys are parameter names and values are parameter settings.
        """
        for i in range(self.grid_size):
            params = {}
            for j, key in enumerate(self.keys):
                idx = int((i // np.prod([len(v) for v in self.param_grid_values[j+1:]])) % len(self.param_grid_values[j]))
                params[key] = self.param_grid_values[j][idx]
            yield params

    def __len__(self):
        """
        Returns the total number of parameter combinations in the grid.
        
        Returns:
            grid_size : int
                The total number of parameter combinations.
        """
        return self.grid_size
