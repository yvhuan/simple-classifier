import matplotlib.pyplot as plt


def plot_list(loss_list=[],
              mystyle='',
              mylabel='',
              mytitle='',
              myxlabel='',
              myylabel='',
              save_path=''
              ):
    plt.plot(loss_list, mystyle, label=mylabel)
    plt.legend()
    plt.title(mytitle)
    plt.ylabel(myxlabel)
    plt.xlabel(myylabel)
    if save_path == '':
        plt.show()
    else:
        plt.savefig(save_path, dip=300)
    plt.close()


if __name__ == '__main__':
    list = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    plot_list(list, 'r^-', 'label', 'title', 'x', 'y')
    plot_list(list, 'y*-', 'label', 'title', 'x', 'y')

