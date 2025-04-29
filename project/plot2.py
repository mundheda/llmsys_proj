import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import json

def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':
    
    train_time_mean = [77.46343839168549, 83.91034734249115]
    train_time_std = [0.0937427282333374, 0.2534443140029907]
    token_ps_mean = [8261.974301070597, 7627.257650507548]
    token_ps_std = [9.998265344999709, 23.03750543501883]
    
    
    plot(train_time_mean,
        train_time_std,
        ['Pipeline Parallel', 'Model Parallel'],
        'pp_vs_mp.png')

    plot(token_ps_mean,
        token_ps_std,
        ['Pipeline Parallel', 'Model Parallel'],
        'pp_vs_mp2.png')