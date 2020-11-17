import matplotlib.pyplot as plt
import numpy as np
import os 

def main():

    percentages = [0.05, 0.1, 0.3, 0.4, 0.7, 0.9]
    p_strings = [f'{p:.2f}' if p < 0.1 else f'{p:.1}' for p in percentages]

    # add 1.0
    percentages.append(1.0); p_strings.append("1.0")
    
    files = [f'results/ssml_omniglot_perc{p}.txt' for p in p_strings]
    ssml_accs = []
    for f_i in files:
        with open(f_i, 'r') as f:
            last_line = f.readlines()[-1]
            ssml_accs.append(float(last_line[11:16]))

    files = [f'results/sl_omniglot_perc{p}.txt' for p in p_strings]
    sl_accs = []
    for f_i in files:
        with open(f_i, 'r') as f:
            last_line = f.readlines()[-1]
            sl_accs.append(float(last_line[11:16]))

    plt.plot([p*100 for p in percentages], ssml_accs, label="SSML Baseline")
    plt.plot([p*100 for p in percentages], sl_accs,   label="SL Baseline")
    plt.legend()
    plt.title('Omniglot (K=1, GAN_epochs=500, MAML_Iterations=5000)')
    plt.xlabel('Percentage of labeles accessible (%)')
    plt.ylabel("Test Accuracy (%")
    plt.savefig('results/results.png')

if __name__ == "__main__":

    main()