import numpy as np
import matplotlib.pyplot as plt

# Define the function
def alpha_2(t,  gamma_0=0.0, gamma_1=20.0):
    gamma_t = - (gamma_0 * t / 2 + t ** 2 / 4 * (gamma_1 - gamma_0))
    return np.exp(2 * gamma_t)

def sigma_2(t,  gamma_0=0.0, gamma_1=20.0):
    return 1 - alpha_2(t, gamma_0, gamma_1)

def lambda_2(t, gamma_0=0.0, gamma_1=20.0):
    lambda_t = np.log(alpha_2(t, gamma_0, gamma_1) / sigma_2(t, gamma_0, gamma_1))
    return lambda_t

def t_(lmd,  gamma_0=0.0, gamma_1=20.0):
    t_lambda = 2 * np.log(np.exp(-2 * lmd) + 1) / (np.sqrt(
        gamma_0 ** 2 + (gamma_1 - gamma_0) * 2 * np.log(np.exp(-2 * lmd) + 1)) + gamma_0)
    return t_lambda

def ve_snr(t,  gamma_0=0.0,  gamma_1=20.0):
    return sigma_2(t,  gamma_0, gamma_1) / alpha_2(t,  gamma_0, gamma_1)

def vp_snr(t,  gamma_0=0.0,  gamma_1=20.0):
    return sigma_2(t,  gamma_0, gamma_1) / alpha_2(t,  gamma_0, gamma_1)

# Define different gamma values
gamma_values = [(0.1, 8.0)]
t_min = 1e-3
t_max = 1

# Plot the function for each set of gamma values
plt.figure(figsize=(10, 6))
for gamma_0, gamma_1 in gamma_values:
    print(gamma_0, gamma_1)
    lambda_min = lambda_2(t_min, gamma_0, gamma_1)
    lambda_max = lambda_2(t_max, gamma_0, gamma_1)
    lambdas = np.random.uniform(lambda_min, lambda_max, size=400)
    t_values = t_(lambdas, gamma_0, gamma_1)
    print(t_values)
    plt.scatter(t_values, lambdas, s=10,
                label=f'gamma_0={gamma_0}, gamma_1={gamma_1}')  # Adjust the `s` parameter for dot size

plt.xlabel('t_values')
plt.ylabel('lambdas')
plt.title('Plot of t_values vs lambdas')
plt.legend()
plt.grid(True)
plt.show()