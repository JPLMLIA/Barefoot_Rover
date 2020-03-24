
## Installing and Running the Barefoot Rover Conda Environment
 The `conda` program is an open source package and environment manager commonly used for Python, and operating at the core of the Anaconda Python distribution.  The YAML file(s) in this directory define a `conda` environment for running the Barefoot Rover python software.

### Installing and Updating Anaconda
Instructions on installing Anaconda are here: https://docs.anaconda.com/anaconda/install/.  Make sure Anaconda's `bin` directory is on your path before proceeding.

If Anaconda is already installed, but is not up to date, you can update it with the following commands (the first updates the `conda` program itself, the latter updates the python packages bundled with Anaconda):
```bash
$ conda update conda
$ conda update anaconda
```

### Enabling Conda on MLIA machines
In order to use the following instruction on an MLIA machine (buffalo, analysis, paralysis), you must first run the following commands:
```bash
$ export PATH=/usr/local/anaconda3/bin:$PATH
$ source /usr/local/anaconda3/etc/profile.d/conda.sh
```

### Creating the Conda Environment
Assuming you have an up-to-date version of anaconda installed, you can install or update the environments with the commands below.

```bash
$ cd $BAREFOOT_ROOT
$ conda env create -f envs/barefoot-env.yml
```

In the above example, the variable `$BAREFOOT_ROOT` is assumed to point to the Barefoot Rover repo root.

### Updating the Conda Environment
If the `barefoot-env.yml` file changes for any reason (e.g. adding new packages), the environment may be updated to be consistent with this new YAML file with the following command.

```bash
$ cd $BAREFOOT_ROOT
$ conda env update -f envs/barefoot-env.yml
```

In the above example, the variable `$BAREFOOT_ROOT` is assumed to point to the Barefoot Rover repo root.

### Usage
With Anaconda's `bin` directory on your path, you can activate the environment by typing `source activate barefoot` (the environment's name, `barefoot`, is defined at the top of the YAML file).  Once active, only the packages, and respective versions, installed into the environment will be available when running Python.

Exiting the environment is a simple `source deactivate` command at the command-line.

```bash
$ source activate barefoot
$ source deactivate
```

### Exporting an Environment

Once created, an environment's definition may be exported to a YAML file again.  The command to do this:
```bash
$ conda env export > environment.yml
```
This `environment.yml` file will contain all packages that were installed, which will be a superset on those originally provided, as it will include any dependencies needed by the packages specified in `barefoot-env.yml`.  It will also contain the exact version of the installed packages, (hopefully) allowing an environment to be perfectly reproduced on another machine.

## Adapted from ONWATCH enviornment documentation written by Steven Lewis.


