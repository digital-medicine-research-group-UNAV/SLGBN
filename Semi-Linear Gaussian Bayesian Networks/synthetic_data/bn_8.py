import numpy as np
import pandas as pd

def generate_bn_8(n = 200):
    
    # Step 1: 
    A = np.random.normal(loc = 0, scale = 1, size = n) # Linear
    B = np.random.normal(loc = 0, scale = 1, size = n) # Linear
    C = np.random.normal(loc = 0, scale = 1, size = n) # Linear
 
    D = np.random.normal(loc = 0.7 * A + 0.6 * B + C * 0.25, scale = 0.4, size = n) # Linear
    E = np.random.normal(loc = np.sin(D) + 0.5 * np.cos(A) + 0.4 * B**2 + 0.6 * np.tanh(C), scale = 0.35, size = n) # Non Linear
    F = np.random.normal(loc = np.log1p(np.abs(D)) + np.exp(0.3 * E), scale = 0.3, size = n) # Non-Linear
    G = np.random.normal(loc = np.cos(F), scale = 0.3, size = n) # Non Linear
    H = np.random.normal(loc = 0.8 * G + 0.6 * F, scale = 0.25, size = n) # Linear
    
    # Combine into a DataFrame
    df = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G, 'H': H})

    return df

def ground_truth_8(df):
    A = df['A'].values
    B = df['B'].values
    C = df['C'].values
    D = df['D'].values
    E = df['E'].values
    F = df['F'].values
    G = df['G'].values
    H = df['H'].values
    
    # Step 1: log p(A)
    logp_A = -0.5*np.log(2*np.pi) - 0.5*np.log(1**2) - 0.5*((A - 0)/1)**2
    
    # Step 2: log p(B)
    logp_B = -0.5*np.log(2*np.pi) - 0.5*np.log(1**2) - 0.5*((B - 0)/1)**2
    
    # Step 3: log p(C)
    logp_C = -0.5*np.log(2*np.pi) - 0.5*np.log(1**2) - 0.5*((C - 0)/1)**2
    
    # Step 4: log p(D | A,B,C)
    mu_D = 0.7*A + 0.6*B + 0.25*C
    logp_D = -0.5*np.log(2*np.pi) - np.log(0.4) - 0.5*((D - mu_D)/0.4)**2
    
    # Step 5: log p(E | A,B,C,D)
    mu_E = np.sin(D) + 0.5*np.cos(A) + 0.4*B**2 + 0.6*np.tanh(C)
    logp_E = -0.5*np.log(2*np.pi) - np.log(0.35) - 0.5*((E - mu_E)/0.35)**2
    
    # Step 6: log p(F | D,E)
    mu_F = np.log1p(np.abs(D)) + np.exp(0.3*E)
    logp_F = -0.5*np.log(2*np.pi) - np.log(0.3) - 0.5*((F - mu_F)/0.3)**2
    
    # Step 7: log p(G | F)
    mu_G = np.cos(F)
    logp_G = -0.5*np.log(2*np.pi) - np.log(0.3) - 0.5*((G - mu_G)/0.3)**2
    
    # Step 8: log p(H | F,G)
    mu_H = 0.8*G + 0.6*F
    logp_H = -0.5*np.log(2*np.pi) - np.log(0.25) - 0.5*((H - mu_H)/0.25)**2
    
    # Joint log-likelihood
    log_lik = logp_A + logp_B + logp_C + logp_D + logp_E + logp_F + logp_G + logp_H
    
    return np.sum(log_lik)