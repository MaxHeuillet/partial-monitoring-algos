# Randomized Confidence Bounds for Partial Monitoring

This is the developpers repository for the project. It contains the scripts to submit experiments on the cluster. This repository contains the implementation of algorithms described in the paper "Randomized Confidence Bounds for Partial Monitoring". For a sandbox codebase with tutorials, please refer to the main branch.

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
   ```

2. **Install Dependencies**:

   ```bash 
   pip install -r requirements.txt
   ```

#### Installation Troubleshooting:

- **Cyipopt**: If you encounter issues installing cyipopt, ensure you have the latest versions of pip, setuptools, and wheel. You may also need additional system dependencies. For more details, see the Cyipopt Installation Guide.
- **Gurobi Alternative**: If you prefer not to use Gurobi, you can use PULP as an alternative optimizer. To do this, install PULP using pip install pulp.

### Running Contextual Experiments

```bash 
   bash benchmark_context_meta.sh
```

### Running Non-contextual Experiments

```bash 
   bash benchmark_nocontext_meta.sh
```

### Acknowledgements

This work was funded through Mitacs with additional support from CIFAR (CCAI Chair). 
We thank Alliance Canada and Calcul Quebec for access to computational resources and staff expertise consultation.
We would like to thank Junpei Komiyama, Taira Tsuchiya, Ian Lienert, Hastagiri P. Vanchinathan and James A. Grant for answering our technical questions and/or providing total/partial access to private code bases of their approaches. We also acknowledge the library pmlib of Tanguy Urvoy that was helpful to implement PM game environments.

