# my_ml_support
The module my_ml_support is a collection of tools for preprocessing of data and for machine learning. It's biult on top of sklearn, matplotlib, pandas and numpy. 

All operations are performed via a Tool instance. A Tool instance is able to keep track of all method calls including the arguments passed to the method. This helps you track how your features engineering and hyperparameter tuning is faring by enabling you to track which configuration produced each result. This feature was very helpful for the work published by Okoli etal (2019).

### References
Okoli, P., Cruz Vega, J., & Shor, R. (2019). Estimating Downhole Vibration via Machine Learning Techniques Using Only Surface Drilling Parameters. In *SPE Western Regional Meeting 2019*. San Jose, California, USA; Society of Petroleum Engineers. Available at: https://doi.org/10.2118/195334-MS.
