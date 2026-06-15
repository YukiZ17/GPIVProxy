# GPIVProxy
Codes for the paper Instrumental and Proximal Causal Inference with Gaussian Processes (GPIV and GPProxy)
To reproduce the experiments, please download the files and run the code **main.py**.

For GPIV on synthetic data, you may change the following configurations directly through the main.py:

1. Number of simulations
2. True functions (log, sine, linear).
3. Single demo (including the demo plots and single ARC) or multiple simulations (mean and standard error of MSE, AUC, Coverage and their plots).
4. Hyperparameter tuning

Other three designs (GPIV on demand, GPProxy on synthetic and demand data) are almost the same except that the true function are fixed.

Some of the ablation study simulations are included in the file GPIVProxyAblation. It is also feasible to change the parameter in the data_generation.py to run some further ablation studies.

