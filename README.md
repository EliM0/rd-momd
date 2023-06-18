# forl-project

Executing the different algorithms:

```bash
python src/evaluation.py --config=<config_dir> --algorithm=['afp', 'dmomd', 'rnn-momd', 'fo-dmomd', 'po-dmomd', 'po-rnn-momd']
```

Check plots:
```bash
tensorboard --logdir=runs
```

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