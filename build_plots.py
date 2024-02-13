import pandas as pd
import matplotlib.pyplot as plt


def plot_csv_files(filenames):
    for filename in filenames:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(filename)

        # Plot the 'Step' column on the x-axis and the 'Value' column on the y-axis
        plt.plot(df["Step"], df["Value"], label=filename)

    # Set the labels and the title for the plot
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Step vs Value")

    # Display the legend
    plt.legend()

    # Show the plot
    plt.show()


# Example usage
csv_files = ["file1.csv", "file2.csv", "file3.csv"]
plot_csv_files(csv_files)
