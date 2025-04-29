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
    
    # read json file
    file_name = "/zfsauton2/home/wentsec/hw/llm_sys/llmsys_s25_hw4/workdir2/wsize1_epoch{epoch}.json"
    train_time = []
    token_ps = []
    for epoch in range(3):
        epoch_file_name = file_name.format(epoch=epoch)
        with open(epoch_file_name, 'r') as f:
            data = json.load(f)
            train_time.append(data['training_time'])
            token_ps.append(data['tokens_per_sec'])
    train_time = np.array(train_time)
    token_ps = np.array(token_ps)
    rn_mean = np.mean(train_time)
    rn_std = np.std(train_time)
    mp_mean = np.mean(token_ps)
    mp_std = np.std(token_ps)
    
    file_name = "/zfsauton2/home/wentsec/hw/llm_sys/llmsys_s25_hw4/workdir/rank{rank}_results_epoch{epoch}.json"
    train_time1 = []
    train_time2 = []
    token_ps = []
    for epoch in range(10):
        epoch_file_name = file_name.format(rank=0, epoch=epoch)
        token_per_sec = 0
        with open(epoch_file_name, 'r') as f:
            data = json.load(f)
            train_time1.append(data['training_time'])
            token_per_sec += data['tokens_per_sec']
        epoch_file_name = file_name.format(rank=1, epoch=epoch)
        with open(epoch_file_name, 'r') as f:
            data = json.load(f)
            train_time2.append(data['training_time'])
            token_per_sec += data['tokens_per_sec']
        token_ps.append(token_per_sec)
    train_time1 = np.array(train_time1)
    train_time2 = np.array(train_time2)
    token_ps = np.array(token_ps)
    mp0_mean = np.mean(train_time1)
    mp0_std = np.std(train_time1)
    mp1_mean = np.mean(train_time2)
    mp1_std = np.std(train_time2)
    pp_mean = np.mean(token_ps)
    pp_std = np.std(token_ps)
    
    print([mp0_mean, mp1_mean, rn_mean])
    print([mp0_std, mp1_std, rn_std])
    print([pp_mean, mp_mean])
    print([pp_std, mp_std])
    
    plot([mp0_mean, mp1_mean, rn_mean],
        [mp0_std, mp1_std, rn_std],
        ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
        'ddp_vs_rn.png')

    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Pipeline Parallel', 'Model Parallel'],
        'pp_vs_mp.png')