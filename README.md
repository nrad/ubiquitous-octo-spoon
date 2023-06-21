# Efficient background generation exercise

This toy simulation demonstrates some of the features of the background from penetrating atmospheric muons in an unsegmented Cherenkov detector like IceCube. While the vast majority of such events can be rejected, their sheer rate means that only the rarest events sneak through into the signal region, making it extremely inefficient to estimate the background rate from unbiased simulation. This toy simulation can be used to mock up strategies for efficiently generating events that will pass a (simplified) signal selection and lead to a less computationally expensive and more precise estimate of the background rate.

## Running in Jupyter

1. Install [jupyter](https://jupyter.org) if it is not already installed in your Python environment:
```
pip install notebook
```
2. Install the dependencies:
```
pip install -r requirements.txt
```
3. Start a Jupyter server, and point your browser at the link printed on the command line:
```
jupyter notebook
```
