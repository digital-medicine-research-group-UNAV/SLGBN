import numpy as np
import pandas as pd
import math

def generate_bn_5(n = 200):

    # Step 1:
    A = np.random.normal(loc=0, scale=1, size=n)
    
    # Step 2:
    B = np.random.normal(loc=0, scale=1, size=n)
    
    # Step 4:
    C = np.random.normal(loc=-0.3 + 0.5*np.sin(A) + 0.5*np.cos(B), scale=0.3, size=n)

    # Step 3:
    D = np.random.normal(loc=np.tanh(C) + np.sin(B), scale=0.2, size=n)
    
    # Step 5:
    E = np.random.normal(loc= 0.6*D + 0.5, scale=0.4, size=n)
    
    # Combine into a DataFrame
    df = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D, 'E': E})

    return df

def ground_truth_5(df):

    A = df['A'].values
    B = df['B'].values
    C = df['C'].values
    D = df['D'].values
    E = df['E'].values
    
    # Step 1: log p(A)
    logp_A = -0.5*np.log(2*np.pi) - 0.5*np.log(1**2) - 0.5*((A - 0)/1)**2
    
    # Step 2: log p(B)
    logp_B = -0.5*np.log(2*np.pi) - 0.5*np.log(1**2) - 0.5*((B - 0)/1)**2
    
    # Step 3: log p(C | A,B)
    mu_C = -0.3 + 0.5*np.sin(A) + 0.5*np.cos(B)
    logp_C = -0.5*np.log(2*np.pi) - np.log(0.3) - 0.5*((C - mu_C)/0.3)**2
    
    # Step 4: log p(D | C,B)
    mu_D = np.tanh(C) + np.sin(B)
    logp_D = -0.5*np.log(2*np.pi) - np.log(0.2) - 0.5*((D - mu_D)/0.2)**2
    
    # Step 5: log p(E | D)
    mu_E = 0.6*D + 0.5
    logp_E = -0.5*np.log(2*np.pi) - np.log(0.4) - 0.5*((E - mu_E)/0.4)**2
    
    # Joint log-likelihood
    log_lik = logp_A + logp_B + logp_C + logp_D + logp_E
    
    return np.sum(log_lik)