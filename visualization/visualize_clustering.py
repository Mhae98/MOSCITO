import matplotlib.pyplot as plt

def show_clustering(labels: list, title: str = None) -> None:
    """
    Show clustering as a plot
    :param labels: List of cluster indexes
    :param title: Title of the plot
    """
    plt.imshow([labels, labels], extent=[0, len(labels), 0, len(labels) / 3], interpolation='nearest')
    plt.yticks([])
    plt.xlabel('Timeframe')
    plt.title(title)
    plt.show()


def save_clustering(labels: list, file_name: str, title: str = None) -> None:
    """
    Save clustering as a plot
    :param labels: List of cluster indexes
    :param file_name: Path and name of the saved plot
    :param title: Title of the plot
    """
    plt.imshow([labels, labels], extent=[0, len(labels), 0, len(labels) / 3])
    plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig(file_name, dpi=400, bbox_inches='tight')
