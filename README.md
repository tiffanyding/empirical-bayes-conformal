# Environment setup

To create the environment and install the necessary packages, run the following commands in Terminal:

    conda create --name conformal_env
    conda activate conformal_env
    conda install --file requirements.txt

Next, to make the environment accessible on Jupyter, make sure the environment is activated, then run

	ipython kernel install --user --name=conformal_env

The name of your new environment is `conformal_env`

# Downloading ImageNet hierarchy
I couldn't find a place to download these files from the ImageNet website, so here are the links I used:
1. https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
1. https://raw.githubusercontent.com/jcjohnson/tiny-imagenet/master/words.txt 
1. https://raw.githubusercontent.com/innerlee/ImagenetSampling/master/Imagenet/data/wordnet.is_a.txt 

# Pipeline (using oracle prior)
1. Run `imagenet_experiments.ipynb` to get oracle prior.
1. Run `get_posterior_quantiles_oracle.ipynb` or `get_posterior_quantiles_oracle.py` to get $\widehat{q}^{EB}$
1. Run `apply_conformal_adjustment.ipynb` (updating `qhat` file path if necessary) to adjust the $\widehat{q}^{EB}$ to guarantee marginal coverage
1. Run `evaluate.ipynb` to evaluate our prediction sets vs. prediction sets of baseline methods