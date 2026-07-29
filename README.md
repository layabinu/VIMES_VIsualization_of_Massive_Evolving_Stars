


https://github.com/user-attachments/assets/30f19727-5089-44c6-8598-965a7b3220e3



Log Default


# VIMES: VIsualization of Massive Evolving Stars
VIMES turns binary stellar evolution simulations into animations. Rapid population synthesis codes like [COMPAS](https://compas.science/) produce detailed output files describing how two stars evolve together over time — exchanging mass, expanding, collapsing — but that numerical output is hard to present intuitively. VIMES reads those files and generates a frame-by-frame animation where the stars' sizes, colors, separation, and eccentricity reflect the actual simulated values at each timestep.

The code works in two steps: a preprocessing stage that maps the simulation data onto evenly-sampled animation frames (with interpolation across large timesteps and equal representation of each evolutionary phase), followed by a rendering stage that produces the animation. There are two visual styles: **default** uses cartoon-like stellar images that change with stellar type, making phase transitions easy to spot; **tulips** derives star colors from effective temperature using the [TULIPS](https://bitbucket.org/elaplace/tulips/src/master/) color conversion ([Laplace et al. 2022](https://ui.adsabs.harvard.edu/abs/2022A%26C....3800516L/abstract)) for a more physically accurate appearance.


## Installation

> **Note on Python versions:** some users have run into an error while building `pygame` when installing VIMES on **Python 3.14**, which is still new and has some package compatibility issues. If you hit this, the recommended fix is to create your environment with **Python 3.13** (e.g. 3.13.14) instead, which works without problems:
>
>     uv tool install --python 3.13 git+https://github.com/layabinu/VIMES_VIsualization_of_Massive_Evolving_Stars.git
>
> or, with pip:
>
>     python3.13 -m venv venv

### Using uv (recommended)
This project is managed by [uv](https://docs.astral.sh/uv/), so the best way to install it is also using uv (follow [install instructions for uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it):

    uv tool install git+https://github.com/layabinu/VIMES_VIsualization_of_Massive_Evolving_Stars.git

The project can also be cloned from Github as usual, and then installed by running the following command in the cloned directory:

    uv tool install -e .

The `-e` flag signals that local changes to the cloned codebase will also automatically be reflected in the installed tool.

### Using pip
If you'd rather just use pip, we recommend installing the cloned project to a virtual environment:

    python -m venv venv
    source venv/bin/activate
    pip install -e .

The project will then need to be run using the virtual environment, which is typically done by activating the environment before running the script.

## Usage
Once installed, the project should be able to be run as follows:

    vimes-preprocess <path-to-input>.h5 <output-path>.npz

In this case, it will take the input HDF5 file and create a frames file at the output path. Once the frames file has been created, we can create the animation by running:

    vimes <path-to-frames>.npz

This also comes with several optional inputs:
 - `--scaling` affects the scaling of the animation. It is `linear` by default, but can also accept `log`.
 - `--images` affects the images that are shown in the animation. The default value is `default`, which renders images of the objects at each stage of evolution, while the other option is `tulips`, which renders circles coloured by their temperature (using the [tulips project](https://bitbucket.org/elaplace/tulips/src/master/)).
 - `--save-mp4` saves the movie at the specified path.
 - `--no-display` stops the animation from being displayed (usually only useful if the animation is also being saved as mp4).

## Code Quality and Testing

If you are developing this project, please run code quality checks before committing to the repository:

```bash
# Format code with Ruff
uv run ruff format src/

# Lint code with Ruff (you can remove the --fix flag to stop it automatically fixing the issues)
uv run ruff check --fix src/
```

It is also recommended to run tests before committing to the repository:

```bash
# Pytest will automatically detect tests in files that looke like test_*.py
uv run pytest
```


https://github.com/user-attachments/assets/7c057ed1-494a-4d30-a08d-a19cd3588de0



Linear Tulips

## Acknowledgement:
VIMES was created by Laya Binu, please contact Laya for any questions.
If you make use of VIMES, we ask you to cite the following Zenodo publication

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20500022.svg)](https://doi.org/10.5281/zenodo.20500022)

@software{laya_binu_2026_20500022,
  author       = {Laya Binu and
                  Floor Broekgaarden},
  title        = {layabinu/VIMES\_VIsualization\_of\_Massive\_Evolving\_S
                   tars: VIMES v1.0.0 — First release
                  },
  month        = jun,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.20500022},
  url          = {https://doi.org/10.5281/zenodo.20500022},
  swhid        = {swh:1:dir:783d47c310e36ce47742fa1da75ec251984fd30e
                   ;origin=https://doi.org/10.5281/zenodo.18503544;vi
                   sit=swh:1:snp:50cf1e46ac092ba078dada7fff8f3c60e74d
                   ff46;anchor=swh:1:rel:1d7eef26bcfa05ed5c068011700c
                   e75114591ba8;path=layabinu-VIMES\_VIsualization\_of\_
                   Massive\_Evolving\_Stars-1bf8443
                  },
}

