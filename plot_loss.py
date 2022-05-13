import matplotlib.pyplot as plt


def plot_loss(loss_list=[], save_path=''):
    plt.plot(loss_list, 'o-', label="Train_Loss")
    plt.legend()
    plt.title("OPT=ADAM, LR=0.005, EPOCHS=10")
    plt.ylabel("VALUE OF LOSS")
    plt.xlabel("iter")
    if save_path == '':
        plt.show()
    else:
        plt.savefig(save_path)


if __name__ == '__main__':
    list = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    plt.plot(list)
    plt.show()