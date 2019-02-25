# Neural Persistence: A Complexity Measure for Deep Neural Networks Using Algebraic Topology 

This repository contains the code for our paper [*Neural Persistence:
A Complexity Measure for Deep Neural Networks Using Algebraic
Topology*](https://openreview.net/pdf?id=ByxkijC5FQ),
which was published as an ICLR 2019 conference paper.

This repository is a work in progress. We aim to add more experiments
over time.

# Deep learning best practices in light of neural persistence 

This repository can be used to reproduce the experiment from Section 4.1
of the publication. To ensure ease of use and reproducibility, it relies
on Docker.

To install Docker, please follow the [official manual](https://www.docker.com/get-started).
Having set up Docker for your operating system, the subsequent sections
guide you through the process.

## Build docker container

```bash
cd $REPODIR
docker build -t neuralpersistence .
```

## Run experiments and summarize results

```bash
docker run -v $PWD/results/:/Neuralpersistence/results/ neuralpersistence python3 -u run_experiments.py
docker run -v $PWD/results/:/Neuralpersistence/results/ neuralpersistence python3 combine_runs.py results/runs/* --output results/combined_runs.csv
```

## Plot the results

```bash
docker run -v $PWD/results/:/Neuralpersistence/results/ neuralpersistence python3 create_plots.py results/combined_runs.csv results/combined_runs.pdf
```

The visualizations of the mean normalized neural persistence, as well as
the test accuracy can be found in `results/combined_runs.pdf`.

# Citation

Please use the following citation to refer to this paper:

    @inproceedings{Rieck19a,
      title     = {Neural Persistence: {A} Complexity Measure for Deep Neural Networks Using Algebraic Topology},
      author    = {Bastian Rieck and Matteo Togninalli and Christian Bock and Michael Moor and Max Horn and Thomas Gumbsch and Karsten Borgwardt},
      booktitle = {International Conference on Learning Representations~(ICLR)},
      year      = {2019},
      url       = {https://openreview.net/forum?id=ByxkijC5FQ},
    }
