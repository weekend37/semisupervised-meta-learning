import matplotlib.pyplot as plt
import numpy as np
import os 

def main():

    percentages = np.arange(0.0,1.0,0.11)
    files = [f'results/ssml_omniglot_perc{p:.1f}.txt' for p in percentages]
    accs = []
    for f_i in files:
        with open(f_i, 'r') as f:
            last_line = f.readlines()[-1]
            accs.append(float(last_line[11:16]))

    plt.plot([p*100 for p in percentages], accs)
    plt.title('Omniglot (K=1, GAN_epochs=500, MAML_Iterations=5000)')
    plt.xlabel('Percentage of labeles accessible (%)')
    plt.ylabel("Test Accuracy (%")
    plt.savefig('results/results.png')

if __name__ == "__main__":

    main()