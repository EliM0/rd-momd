# Installing Dependencies
To run the code you first need to install the required dependencies and apply
our OpenSpiel patch. To install most of the dependencies, a Conda
[environment.yml](./environment.yml) is provided. The following steps are
required to install and patch OpenSpiel.

1. Clone the [OpenSpiel GitHub repository](https://github.com/deepmind/open_spiel/).
2. Run the [`install.sh`](https://github.com/deepmind/open_spiel/blob/master/install.sh) to install OpenSpiel's dependencies.
3. Apply our OpenSpiel patch by running `git am <path-to-project>/open_spiel_patches/open_spiel.patch` from the root of the OpenSpiel repository.
4. Create a build directory and build OpenSpiel by running 
   ```bash
   mkdir build && cd build
   cmake -DPython3_EXECUTABLE=$(which python3) -DCMAKE_CXX_COMPILER=$(which clang++) ../open_spiel
   make -j12
   ```
5. Add the built OpenSpiel to your Python PATH using
   ```bash
   export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel>
   export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel>/build/python
   ```

# Experiments

## Algorithms

In this repository, we have population-dependent versions of D-MOMD and D-AFP. These algorithms, can be run with full observability of the population or knowledge of the population of the agent's neighbour (partial observability). We also developed a population-dependent algorithm that takes into account the previous history of the agent to solve MFGs.


|           |                 Algorithm                  | basic id |    fully observable id     |  partially observable id   |
|:----------|:------------------------------------------:|:--------:|:--------------------------:|:--------------------------:|
|   D-AFP   |        Deep Average Fictitious Play        |   dafp   |            -               |            -               |
|   D-MOMD  |   Deep Munchausen Online Mirror Descent    |  dmomd   |            -               |            -               |
|  D-AFPP   |        Population-dependent D-AFP          |    -     |          fo-dafp           |          po-dafp           |
|  D-MOMDP  |        Population-dependent D-MOMD         |    -     |          fo-dmomd          |          po-dmomd          |
|  RD-MOMD  | Recurrent Munchausen Online Mirror Descent | rnn-momd | fo-1rd-momd / fo-trd-momd* | po-1rd-momd / po-trd-momd* |

\* In 1rd-momd, the algorithm samples trajectories of length 1. In trd-momd, the algorithm samples trajectories of length $T$ were $T$ is the horizon.

## Running different experiments

To run different algorithms, execute inside the root folder of the repo the following command:

```bash
python src/evaluation.py --config=<config_file_path> --algorithm=<algorithm_id> --logdir=<log_dir_path>
```

If the configuration file or the log directory are not specified, the code will run the experiments in the 2D Crowd Environment from OpenSpiel and it will save the logs into the `runs` directory.

For example, to run an experiment on D-MOMD in the Four Rooms OpenSpiel MFGs environment with partial observability, execute:

```bash
python src/evaluation.py --config=4rooms_config.yml --algorithm=po-dmomd
```

To check the results generated, simply execute:

```bash
tensorboard --logdir=<log_dir_path>
```