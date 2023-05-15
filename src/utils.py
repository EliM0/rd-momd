import yaml
import matplotlib.pyplot as plt

def read_config(config_file):
    with open(config_file, mode='r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
        return cfg
    
def plot_explotability(fp, omd, momd, rnnmomd, title):
    plt.figure()
    plt.title(title)

    plt.xlabel("Iterations")
    plt.ylabel("Exploitability")
    
    plt.semilogy(fp)
    plt.semilogy(omd)
    plt.semilogy(momd)
    plt.semilogy(rnnmomd)

    plt.grid(True)
    plt.legend(["Fictitious Play", "Online Mirror Descent", "Munchausen OMD", "RNN + MOMD"])
    plt.show()
