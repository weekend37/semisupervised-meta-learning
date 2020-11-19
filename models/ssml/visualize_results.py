import matplotlib.pyplot as plt
import numpy as np
import os 

def main():

    folder = "results_K1"
    percentages = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.7, 0.9]
    p_strings = [f'{p:.2f}' if p < 0.1 else f'{p:.1}' for p in percentages]
    percentages.append(1.0); p_strings.append("1.0") # add 1.0
    
    # SEMI SUPERVISED BASELINE
    files = [f'{folder}/ssml_omniglot_perc{p}.txt' for p in p_strings]
    ssml_accs = []
    for f_i in files:
        with open(f_i, 'r') as f:
            last_line = f.readlines()[-1]
            ssml_accs.append(float(last_line[11:16]))

    # SUPERVISED BASELINE
    p_strings[0] = "0.001" # proxy for 0% in SL is 0.001%
    files = [f'{folder}/sl_omniglot_perc{p}.txt' for p in p_strings]
    sl_accs = []
    for f_i in files:
        with open(f_i, 'r') as f:
            last_line = f.readlines()[-1]
            sl_accs.append(float(last_line[11:16]))

    # OUR METHOD
    ours_accs = [ssml_accs[0], 50.44, 78.52, 85.28, sl_accs[-1]]
    ours_perc = [0, 0.05, 0.2, 0.5, percentages[-1]]

    # Unsupervised
    usl_acc = [ssml_accs[0]]*2
    usl_perc = [0,1]

    plt.plot([p*100 for p in usl_perc], usl_acc, label="Unsupervised", linestyle='dashed', color="green")
    plt.plot([0,100], [sl_accs[-1]]*2, label="Supervised", linestyle='dashed', color="red")

    # plt.plot([p*100 for p in percentages], sl_accs,   label="Supervised",   color="red")
    plt.plot([p*100 for p in percentages], ssml_accs, label="Semi-Supervised (naive)", color="orange")
    plt.plot([p*100 for p in ours_perc], ours_accs,   label="Semi-Supervised (ours)",  color="blue")
    plt.legend()
    plt.title('Omniglot (K=1, N=5, GAN_epochs=500, MAML_Iterations=5000)')
    plt.xlabel('Percentage of labels accessible (%)')
    plt.ylabel("Test Accuracy (%)")
    plt.savefig(folder+'/results.png')

if __name__ == "__main__":

    main()