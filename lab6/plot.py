import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_curve(name, steps, ewma_reward_list, total_reward_list):
    plt.figure(figsize=(15,4))
    plt.title(f'Result Comparison')
    plt.plot(steps, total_reward_list, color='royalblue', label='Total reward')
    plt.xlabel('steps')
    plt.ylabel('reward')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'./images/{name}_total_reward.png')
    plt.show()

    plt.figure(figsize=(15,4))
    plt.title(f'Result Comparison')
    plt.plot(steps, ewma_reward_list, color='limegreen', label='Ewma reward')
    plt.xlabel('steps')
    plt.ylabel('reward')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'./images/{name}_ewma_reward.png')
    plt.show()