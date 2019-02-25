# Neural Persistence

This repository can be used to reproduce some experiments presented in the publication.
To ensure ease of use and reproducibility, it relies on docker which can be installed following the instructions at https://www.docker.com/community-edition .

## Build docker container

```bash
cd $REPODIR
docker build -t neuropersistence .
```

## Run experiments and summarize results

```bash
docker run -v $PWD/results/:/Neuropersistence/results/ neuropersistence python3 -u run_experiments.py
docker run -v $PWD/results/:/Neuropersistence/results/ neuropersistence python3 combine_runs.py results/runs/* --output results/combined_runs.csv
```

## Plot results

```bash
docker run -v $PWD/results/:/Neuropersistence/results/ neuropersistence python3 create_plots.py results/combined_runs.csv results/combined_runs.pdf
```

Visualizations of the mean normalized neural persistence and test accuracy can afterwards be found in `results/combined_runs.pdf`
