from enum import unique
from textwrap import wrap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, skew, kurtosis, zscore
import statistics
import math
class Stat:
    """
    
    """
    def __init__(self):
        self.data = []

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

    def frequency_table(self, class_interval, class_count):
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

    def histogram(self):
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
        mu, sigma = norm.fit(self.data)
        x = np.linspace(min(self.data), max(self.data), 100)
        y = norm.pdf(x, mu, sigma)
        plt.plot(x, y, color='blue')
        plt.xlabel('X')
        plt.ylabel('Probability Density')
        plt.title('Distribution Curve')
        plt.show()

    def stem_and_leaf(self):
        if not self.data:
            return print("No data available. Please enter data first.")
        stem_leaf_dict = {}
        for value in self.data:
            stem = value // 10
            leaf = value % 10
            if stem in stem_leaf_dict:
                stem_leaf_dict[stem].append(leaf)
            else:
                stem_leaf_dict[stem] = [leaf]
        print("Stem| Leaf")
        for stem, leaves in sorted(stem_leaf_dict.items()):
            print(f"{stem} | {' '.join(map(str, sorted(leaves)))}")

    def box_plot(self):
        if not self.data:
            return "No data available. Please enter data first."            
        plt.boxplot(self.data)
        plt.xlabel('Data')
        plt.ylabel('Values')
        plt.title('Box Plot')
        plt.show()

    def bar_graph(self):
        if not self.data:
            return "No data available. Please enter data first."
        unique_values, value_counts = np.unique(self.data, return_counts=True)
        plt.bar(unique_values, value_counts)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Bar Graph')
        plt.show()
    
    def pie_chart(self):
        if not self.data:
            return "No data available. Please enter data first."
        unique_values, value_counts = np.unique(self.data, return_counts=True)
        plt.pie(value_counts, labels=unique_values, autopct='%1.1f%%')
        plt.title('Pie Chart')
        plt.show()
            
    def arithmetic_mean(self):
        mean = statistics.mean(self.data)
        print(f'Mean:{mean}') 
        return mean
    
    def median(self):
        median = statistics.median(self.data)
        print(f'Median: {median}')
        return median
    
    def mode(self):
        mode = statistics.mode(self.data)
        print('Mode:', mode)                 
        return mode 
    
    def compare_amm(self):
        mean = self.arithmetic_mean()
        median = self.median()
        mode = self.mode()
        if mode >= median and median >= mean:
            print('Left skewed  & negatively biased distribution')
        elif mean >= median and median >= mode:
            print('Right skewed & positively biased distribution')   
        elif mean == median == mode:
            print('Symmetrical distribution')    
        
    def geometric_mean(self):
        if not self.data:
            return print('No data avaible. Please enter data first.')
        product = 1
        for value in self.data:
            product *= value
        geometric_mean = math.pow(product, 1/len(self.data))        
        print(f'Geometric mean: {geometric_mean}')
        return geometric_mean

    def predict_future_value(self):
        if not self.data:
            return print('No data available. Please enter data first.')
        """
        This formula is used to calculate the final value of an initial value after a certain number of years with a specific growth rate.
        Po: Initial value
        r: Growth rate
        n: Number of years or time interval
        """
        Po = float(input("Enter the initial value (Po): "))
        r = float(input("Enter the growth rate (r): "))
        n = float(input("Enter the number of years (n): "))
        Pn = Po * (1 + r)**n 
        print(f'The final value (Pn) after {n} years will be: {Pn}')
        return Pn

    def harmonic_mean(self):
        if len(self.data) == 0:
            return None
        reciprocal_sum = sum(1 / num for num in self.data)
        harmonic_mean = len(self.data) / reciprocal_sum
        print(f'Harmonic Mean: {harmonic_mean}')
        return harmonic_mean

    def weighted_average(self):
        if not self.data:
            return print('No data available. Please enter data first.')
        self.weights = []
        for i in range(len(self.data)):
            weight = float(input(f"Enter the weight for data entry {i + 1}: "))
            self.weights.append(weight)
        if len(self.data) != len(self.weights):
            return print('Number of data entries and weights should be the same.')
        weighted_sum = sum(x * w for x, w in zip(self.data, self.weights))
        total_weight = sum(self.weights)
        weighted_avg = weighted_sum / total_weight
        print(f'Weighted Average: {weighted_avg}')
        return weighted_avg

    def squared_mean(self):
        if not self.data:
            return print('No data available. Please enter data first.')
        squared_values = [x**2 for x in self.data]
        squared_mean = math.sqrt(sum(squared_values) / len(self.data))
        print(f'Squared Mean: {squared_mean}')
        return squared_mean

    def percentile(self, p):
        percentile_value = np.percentile(self.data, p)
        print(f'{p}th percentile: {percentile_value}')
        return percentile_value
    
    def quartiles(self):
        Q1 = np.percentile(self.data, 25)
        Q2 = np.percentile(self.data, 50)
        Q3 = np.percentile(self.data, 75)
        print(f'Q1: {Q1}\nQ2: {Q2}\nQ3: {Q3}')
        return Q1, Q2, Q3
    
    def ceyrek_sapma(self):
        Q1, Q3, Q2 = self.quartiles()
        ceyrek_sapma = (Q3 - Q1) / 2
        print(f'Ceyrek Sapma: {ceyrek_sapma}')
        return ceyrek_sapma

    def average_absolute_deviation(self):
        if not self.data:
            return print('No data available. Please enter data first.')

        avd = np.mean(np.abs(self.data - np.mean(self.data)))
        print(f'Average absolute deviation: {avd}')
        return avd

    def variance(self):
        if not self.data:
            return print('No data available. Please enter data first.')
    
        variance = np.var(self.data)
        print(f'Variance: {variance}')
        return variance

    def standard_deviation(self):
        if not self.data:
            return print('No data available. Please enter data first.')
    
        standard_deviation = np.std(self.data)
        print(f'Standard Deviation: {standard_deviation}')
        return standard_deviation

    def standard_error(self):
        if not self.data.size:
            return print('No data available. Please enter data first.')

        standard_error = np.std(self.data) / np.sqrt(len(self.data))
        print(f'Standard Error: {standard_error}')
        return standard_error

    def calculate_coefficient_of_variation(self):
        if not self.data.size:
            return print('No data available. Please enter data first.')

        mean = np.mean(self.data)
        std_deviation = np.std(self.data)
        coefficient_of_variation = (std_deviation / mean) * 100
        print(f'Coefficient of Variation: {coefficient_of_variation}%')
        return coefficient_of_variation
    
    def skewness(self):
        if not self.data:
            return print('No data available. Please enter data first.')
        skewness_value = skew(self.data)
        print(f'Skewness: {skewness_value}')
        return skewness_value
    
    def kurtosis(self):
        if not self.data:
            return print('No data available. Please enter data first.')
        kurtosis_value = kurtosis(self.data)
        print(f'Kurtosis: {kurtosis_value}')
        return kurtosis_value

    def analyze_outliers(self):
        if len(self.data) == 0:
            print("Veri yok. Lütfen veri ekleyin.")
            return

        z_scores = zscore(self.data)
        threshold = 3

        outliers = np.where(np.abs(z_scores) > threshold)[0]

        if len(outliers) == 0:
            print("Aykırı Değer Bulunamadı.")
        else:
            print("Aykırı Değerler:")
            print("-----------------")
            for index in outliers:
                print(self.data[index])
            print("-----------------")

        plt.boxplot(self.data, notch=True, vert=False)
        plt.title("Box Plot")
        plt.xlabel("Değerler")
        plt.ylabel("Box Plot")
        plt.show()
        

        


# Test
stat = Stat()
stat.get_data_from_csv()

     

