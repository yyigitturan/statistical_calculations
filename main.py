import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

class Stat:
    def __init__(self):
        self.data = []

    # Other functions will be added here...

    def get_data_from_csv(self):
        while True:
            csv_file = input("Enter the CSV file name: ")
            try:
                data = pd.read_csv(csv_file)
                columns = data.columns
                print("Columns in the CSV file:", columns)
                column_name = input("Enter the column name to retrieve data: ")
                if column_name in columns:
                    self.data = data[column_name].values.flatten().tolist()
                    break
                else:
                    print("Invalid column name. Please try again.")
            except Exception as e:
                print("Error occurred while reading the CSV file:", e)

    def get_data_manually(self):
        while True:
            num = input("Enter a number (to quit, enter 'q'; to delete the last entry, enter 'd'): ")
            if num.lower() == 'q':
                break
            elif num.lower() == 'd':
                if self.data:
                    self.data.pop()
                    print("Last entry deleted.")
                else:
                    print("No data available to delete.")
            else:
                try:
                    num = float(num)
                    self.data.append(num)
                except ValueError:
                    print("Invalid input. Please try again.")

    def generate_frequency_table(self, class_interval, class_count):
        if not self.data:
            print("No data available. Please enter data first.")
            return

        min_value = min(self.data)
        max_value = max(self.data)
        limits = [min_value]
        for i in range(1, class_count + 1):
            limits.append(min_value + i * class_interval)

        class_midpoints = [(limits[i] + limits[i + 1]) / 2 for i in range(len(limits) - 1)]

        class_frequencies = [0] * class_count
        cumulative_frequencies = [0] * class_count
        total_frequency = 0
        for value in self.data:
            for i in range(class_count):
                if value >= limits[i] and value < limits[i + 1]:
                    class_frequencies[i] += 1
                    total_frequency += 1
                    cumulative_frequencies[i] = total_frequency
                    break

        class_percentages = [(fi / total_frequency) * 100 for fi in class_frequencies]

        table = pd.DataFrame({
            'Lower Limit': limits[:-1],
            'Upper Limit': limits[1:],
            'Class Midpoints': class_midpoints,
            'fi': class_frequencies,
            'Fi': cumulative_frequencies,
            'pi': class_percentages
        })

        return table

    def arithmetic_mean(self):
        if self.data == None:
            print("No data available.")
        else:
            return sum(self.data) / len(self.data)

    def generate_histogram(self):
        if not self.data:
            print("No data available. Please enter data first.")
            return

        plt.hist(self.data, bins='auto', edgecolor='black')
        plt.xlabel('X')
        plt.ylabel('Frequency')
        plt.title('Histogram')
        plt.show()

    def plot_frequency_polygon(self):
        if not self.data:
            print("No data available. Please enter data first.")
            return

        # Calculate class midpoints and frequencies for the data
        class_counts, class_bins, _ = plt.hist(self.data, bins='auto', edgecolor='black')
        class_midpoints = [(class_bins[i] + class_bins[i + 1]) / 2 for i in range(len(class_bins) - 1)]
        frequencies = class_counts.tolist()

        # Plot the frequency polygon
        plt.plot(class_midpoints, frequencies, marker='o', linestyle='-', color='blue')
        plt.xlabel('Class Midpoints')
        plt.ylabel('Frequencies')
        plt.title('Frequency Polygon')
        plt.show()

    def plot_distribution_curve(self):
        if not self.data:
            print("No data available. Please enter data first.")
            return

        # Fit a normal distribution curve to the data
        mu, sigma = norm.fit(self.data)

        # Create an array of values to generate the curve
        x = np.linspace(min(self.data), max(self.data), 100)
        y = norm.pdf(x, mu, sigma)

        # Plot the distribution curve
        plt.plot(x, y, color='blue')
        plt.xlabel('X')
        plt.ylabel('Probability Density')
        plt.title('Distribution Curve')
        plt.show()

    def generate_stem_and_leaf(self):
        if not self.data:
            print("No data available. Please enter data first.")
            return

        stem_leaf_dict = {}
        for value in self.data:
            stem = value // 10
            leaf = value % 10
            if stem in stem_leaf_dict:
                stem_leaf_dict[stem].append(leaf)
            else:
                stem_leaf_dict[stem] = [leaf]

        for stem, leaves in sorted(stem_leaf_dict.items()):
            print(f"{stem} | {' '.join(map(str, sorted(leaves)))}")

# Usage example:
stat = Stat()
stat.get_data_manually()
stat.generate_stem_and_leaf()




    
