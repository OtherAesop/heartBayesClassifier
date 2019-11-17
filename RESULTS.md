# Test Result Summary

## Abstract

In this experiment we implemented a naive bayes classifier that uses a continuous distribution to measure conditional
probability. Here we attempt to analyze 'samples' of abnormal heart data to predict whether or not it is abnormal based
on feature analysis. We perform the experiment on 4 unique sets of data, `spect-orig`, `spect-itg`, `spect-resplit`, and 
`spect-resplit-itg` in that order.

## How to build and run

Clone to an IDE of your choice and run `main.py` to run all experiments and produce results organized in the above fashion.

## Results

The resulting accuracy scores are as follows
**Orig**\
Accuracy= 127/187,   67.9%.\
**Orig-itg**\
Accuracy= 139/187,   74.3%.\
**Resplit**\
Accuracy= 65/90,   72.2%.\
**Resplit-itg**\
Accuracy= 61/90,   67.8%.\

Our detailed results are as follows\
**Orig**\
True Positives = 116, False Positives = 4, True Negatives = 11, False Negatives = 56\
Precision= 116/120,   96.7%.\
Recall= 116/172,   67.4%.\
FP Rate= 4/15,   26.7%.\
Accuracy= 127/187,   67.9%.\
**Orig-itg**\
True Positives = 125, False Positives = 1, True Negatives = 14, False Negatives = 47\
Precision= 125/126,   99.2%.\
Recall= 125/172,   72.7%.\
FP Rate= 1/15,    6.7%.\
Accuracy= 139/187,   74.3%.\
**Resplit**\
True Positives = 47, False Positives = 1, True Negatives = 18, False Negatives = 24\
Precision= 47/48,   97.9%.\
Recall= 47/71,   66.2%.\
FP Rate= 1/19,    5.3%.\
Accuracy= 65/90,   72.2%.\
**Resplit-itg**\
True Positives = 45, False Positives = 3, True Negatives = 16, False Negatives = 26\
Precision= 45/48,   93.8%.\
Recall= 45/71,   63.4%.\
FP Rate= 3/19,   15.8%.\
Accuracy= 61/90,   67.8%.\

## Hardware and Software

My old laptop and IntelliJ IDEA

## Notes

Running Bayesian learning with continuous data modeling is a bit different from discrete as you cannot make use of 
Laplace smoothing or m-estimation and instead must rely on Bessel's Correction (easily done through numpy.) This means
that you are not able to directly observe and prevent 0% probabilities and using `log` to prevent underflow is much more
difficult than it should be and doesn't seem to have the potential to yield any serious performance gains.