import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import norm
import torch

def compute_auc_arc(removed_ratios, accurate_ratios):
    x = np.asarray(removed_ratios)
    y = np.asarray(accurate_ratios)
    if len(x) == 0:
        return 0.0
    return trapezoid(y, x)

def compute_mean_ci_width(gp, x_test, confidence=0.95):
    z_score = norm.ppf((1 + confidence) / 2)
    widths = []
    for x in x_test:
        _, var = gp.predict(x)
        std = torch.sqrt(torch.clamp(var, min=1e-10))
        width = 2 * z_score * std.item()
        widths.append(width)
    return np.mean(widths)