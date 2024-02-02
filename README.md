# Randomized Confidence Bounds for Partial Monitoring

This repository contains the implementation of algorithms described in the paper "Randomized Confidence Bounds for Partial Monitoring".

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.8
- pip

### Installation

Follow these steps to set up your environment and run the experiments:

1. **Create a Virtual Environment**:
   ```bash
   python -m venv env
   source env/bin/activate  

2. **Install Dependencies**:

   ```bash 
   pip install -r requirements.txt
   ```

#### Installation Troubleshooting:

- **Cyipopt**: If you encounter issues installing cyipopt, ensure you have the latest versions of pip, setuptools, and wheel. You may also need additional system dependencies. For more details, see the Cyipopt Installation Guide.
- **Gurobi Alternative**: If you prefer not to use Gurobi, you can use PULP as an alternative optimizer. To do this, install PULP using pip install pulp.

### Running Experiments

- **Non-contextual Experiments**: To run non-contextual experiments, use the Jupyter notebook experiment_noncontextual.ipynb.
- **Contextual Experiments**: For contextual experiments, refer to the experiment_contextual.ipynb notebook.
- **Use case Experiments**: For the use case, refer to the Use_case folder, approaches C-CBP, C-RandCBP  and ExploreFully are in the utils.py file. 

### Acknowledgements

TBD


