# Environment setup

To create the environment and install the necessary packages, run the following commands in Terminal:

    conda create --name conformal_env
    conda activate conformal_env
    conda install --file requirements.txt

Next, to make the environment accessible on Jupyter, make sure the environment is activated, then run

	ipython kernel install --user --name=conformal_env

The name of your new environment is `conformal_env`