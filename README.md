Machine Unlearning under Overparameterization
=================================================

Official implementation of the paper:
**"Machine Unlearning under Overparameterization" (NeurIPS 2025)**
Paper: https://arxiv.org/abs/2505.22601

This repository contains the code for reproducing all experiments and results from the paper.

---

### Repository Structure

- **common/** and **experiments/** — source code for all experiments  
- **configs/** — JSON configuration files defining hyperparameters  
- **scripts/** — bash scripts to launch experiments  

The file **common/grid_search.py** manages parameter sweeps for all methods.  
It supports both single-node and distributed execution via MPI.  
To enable multi-node mode, pass the `--mpi` flag when running `common/grid_search.py`.  
Without this flag, the script will run in single-node mode by default.

---

### Dependencies

Install core dependencies using:

```bash
pip install -r requirements.txt
```

---

### Citation

```bibtex
@inproceedings{unlearningUnderOverparameterization,
  title={Machine Unlearning under Overparameterization},
  author={Jacob L. Block and Aryan Mokhtari and Sanjay Shakkottai},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

---

### License

This project is licensed under the **Apache 2.0 License** (see `LICENSE` file).
