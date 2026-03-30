import matplotlib.pyplot as plt


def plot_bin_counts(
    x,
    y,
    bin_range=(0, 1000)
):
    """
    Args:
        x (array): downsampled or predicted counts per bin
        y (array): true counts per bin
        bin_range (tuple): range of bin indices to plot
    """
    
    fig, ax = plt.subplots()
    sub_x = x[bin_range[0]:bin_range[1]]
    sub_y = y[bin_range[0]:bin_range[1]]
    bins = [i for i in range(bin_range[0],bin_range[1])]
    ax.bar(x = bins,
           height = (sub_y),
           color = 'blue',
           alpha = 0.8,
           label = 'Full Counts')
    ax.bar(x = bins,
            height = (sub_x),
            color = 'red',
            alpha = 0.8,
           label = 'Sparse Counts')
    ax.set(xlabel = 'Chr19 Bin Index',
           ylabel = 'Counts in Bin')
    
    ax.legend()

    return fig

def plot_bin_counts_single(
    x,
    bin_range=(0, 1000)
):
    """
    Args:
        x (array): downsampled or predicted counts per bin
        y (array): true counts per bin
        bin_range (tuple): range of bin indices to plot
    """
    
    fig, ax = plt.subplots()
    sub_x = x[bin_range[0]:bin_range[1]]
    bins = [i for i in range(bin_range[0],bin_range[1])]
    ax.bar(x = bins,
            height = (sub_x),
            color = 'red',
            alpha = 0.8,)
    ax.set(xlabel = 'Chr19 Bin Index',
           ylabel = 'Counts in Bin')
    
    ax.legend()

    return fig