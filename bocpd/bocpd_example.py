# ------------------------------------------------------------------------------------------------ #
# bocpd_example.py
# Author: Alexandre Rodrigues Emidio [alexandre dot rodrigues dot emidio at gmail dot com]
#
# Python version: 2.7
#
# Software description:
# TODO: description...
# ------------------------------------------------------------------------------------------------ #

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


if __name__ == '__main__':

    # model definition
    # x_i ~ Normal(mu, sigma2)
    # mu ~ Normal(mu0, sigma2/k)
    # 1/sigma2 ~ Gamma(alpha, beta)

    # dataset parameter and model hyperparameters
    n = 1000
    x = np.empty(shape=(n, ))
    t = np.linspace(0, x.shape[0], num=x.shape[0])
    chg_points = []
    mu_0 = 0.0
    k_0 = 1.0
    alpha_0 = 1.0
    beta_0 = 1.0
    l = 200.0    # hazard function parameter (l > 1)
    # np.random.seed(3219)    # set the seed for the random generator

    # generate some data according to the model
    for i, _ in enumerate(x):
        rand = np.random.uniform()
        if rand < 1/l or i == 0:    # event/changepoint occurred
            if i != 0:
                chg_points.append(i)
            sigma2 = 1.0 / np.random.gamma(alpha_0, 1.0 / beta_0)
            mu = np.random.normal(mu_0, np.sqrt(sigma2 / k_0))
        x[i] = np.random.normal(mu, np.sqrt(sigma2))

    # plot dataset (sampled points and changepoint locations)
    plt.figure()
    plt.title('Synthetic Dataset', fontsize=16)
    plt.xlabel(r'$t$', fontsize=16)
    plt.ylabel(r'$x(t)$', fontsize=16)
    plt.plot(t, x, 'bo')
    [plt.axvline(chg_point, color='r', ls='dashed') for chg_point in chg_points]
    plt.grid(True)
    plt.show()

    # BOCPD algorithm
    p_r = np.zeros(shape=(n + 1, n))    # allocate memory to store the entries of p(r_t|x_{1:t})
    map_p_r = np.empty(shape=(n, ))    # allocate memory to store the MAP estimate of p(r_t|x_{1:t})

    # BOCPD initialization
    k_n = np.array([k_0])
    alpha_n = np.array([alpha_0])
    mu_n = np.array([mu_0])
    beta_n = np.array([beta_0])
    x_sum = np.array([0.0])
    x2_sum = np.array([0.0])
    p_r_x = np.array([1.0])    # p(r_t, x_{1:t}), t = 0 := p(r_0 = 0) = 1

    # start BOCPD loop
    for i, x_i in enumerate(x):    # observe new datum
        # compute the predictive probabilities p(x_t|r_{t-1}, x_{t-r:t-1})
        p_x = student_t.pdf(x_i, 2.0*alpha_n, mu_n, np.sqrt(beta_n*(k_n + 1.0)/(alpha_n*k_n)))

        # compute the growth probabilities p(r_t != 0, x_{1:t})
        p_rx_x = (1.0 - 1.0/l)*p_x*p_r_x

        # compute the changepoint probability, p(r_t = 0, x_{1:t})
        p_r0_x = (1.0/l)*np.dot(p_x, p_r_x)

        # update the probability distribution p(r_t, x_{1:t}) and normalize it to obtain
        # p(r_t|x_{1:t})
        p_r_x = np.append(p_r0_x, p_rx_x)
        p_r_x = p_r_x/np.sum(p_r_x)

        # keep the result in memory
        p_r[0:i+2, i] = p_r_x    # p(r_t|x_{1:t})
        map_p_r[i] = p_r_x.argmax()    # argmax r_t p(r_t|x_{1:t})

        # update sufficient statistics
        x_sum = np.append(0.0, x_sum + x_i)
        x2_sum = np.append(0.0, x2_sum + x_i**2)
        k_n = np.append(k_0, k_n + 1.0)
        alpha_n = np.append(alpha_0, alpha_n + 0.5)
        mu_n = (k_0*mu_0 + x_sum)/k_n
        beta_n = beta_0 + 0.5*(mu_0**2*k_0 + x2_sum - mu_n**2*k_n)

    # plot the result (synthetic data and BOCPD result)
    ax1 = plt.subplot(2, 1, 1)
    plt.title('BOCPD result', fontsize=16)
    plt.ylabel(r'$x(t)$', fontsize=16)
    plt.plot(t, x, 'bo')
    [plt.axvline(chg_point, color='r', ls='dashed') for chg_point in chg_points]
    plt.grid(True)
    plt.subplot(2, 1, 2, sharex=ax1)
    plt.xlabel(r'$t$', fontsize=16)
    plt.ylabel(r'$r_t$', fontsize=16)
    plt.plot(t, map_p_r, 'r', label=r'$argmax_{r_t}$' + ' ' + r'$p(r_t|x_{1:t})$')
    plt.imshow(-np.log(p_r), origin='lower', cmap=plt.cm.gray, aspect='auto')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
