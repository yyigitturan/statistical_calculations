import pandas as pd
import statistics
import numpy as np
class Statistics:
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

    def arithmetic_mean(self):
        if not self.data:
            print("No data available.")
            return

        mean = sum(self.data) / len(self.data)
        print("Arithmetic mean:", mean)

    def weighted_mean(self):
        if not self.data:
            print("No data available.")
            return

        weights = []
        while True:
            weight = input("Enter a weight for the corresponding number (to finish, enter 'q'): ")
            if weight.lower() == 'q':
                break
            try:
                weight = float(weight)
                weights.append(weight)
            except ValueError:
                print("Invalid input. Please try again.")

        if len(self.data) != len(weights):
            print("Number of values and weights should be the same.")
            return

        weighted_sum = sum([num * weight for num, weight in zip(self.data, weights)])
        total_weight = sum(weights)
        weighted_mean = weighted_sum / total_weight

        print("Weighted mean:", weighted_mean)

    def calculate_median(self):
        if not self.data:
            print("No data available.")
            return

        median = statistics.median(self.data)
        print("Median:", median)

    def calculate_percentile(self):
        if not self.data:
            print("No data available.")
            return

        while True:
            try:
                p = float(input("Enter the percentile value (0-100): "))
                if 0 <= p <= 100:
                    break
                else:
                    print("Invalid percentile value. Please enter a value between 0 and 100.")
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

        percentile = np.percentile(self.data, p)
        print(f"{p}th percentile:", percentile)

    def weighted_median(self):
        if not self.data:
            print("No data available.")
            return

        weights = []
        while True:
            weight = input("Enter a weight for the corresponding number (to finish, enter 'q'): ")
            if weight.lower() == 'q':
                break
            try:
                weight = float(weight)
                weights.append(weight)
            except ValueError:
                print("Invalid input. Please try again.")

        if len(self.data) != len(weights):
            print("Number of values and weights should be the same.")
            return

        sorted_indices = np.argsort(self.data)
        sorted_data = np.array(self.data)[sorted_indices]
        sorted_weights = np.array(weights)[sorted_indices]

        cumulative_weights = np.cumsum(sorted_weights)
        total_weight = np.sum(sorted_weights)

        midpoint = total_weight / 2.0

        if np.all(cumulative_weights <= midpoint):
            return sorted_data[-1]

        index = np.searchsorted(cumulative_weights, midpoint, side='right')
        if cumulative_weights[index - 1] == midpoint:
            return (sorted_data[index - 1] + sorted_data[index]) / 2.0

        return sorted_data[index]
    def perform_operations(self):
        while True:
            print("\nOperations Menu:")
            print("1. Enter data manually")
            print("2. Import data from CSV")
            print("-------------------------")
            print("3. Calculate arithmetic mean")
            print("4. Calculate weighted mean")
            print("5. Calculate median")
            print("6. Calculate percentile")
            print("7. Calculate weighted median")
            print("100. Exit")

            choice = input("Enter your choice: ")
            if choice == '1':
                self.get_data_manually()
            elif choice == '2':
                self.get_data_from_csv()
            elif choice == '3':
                self.arithmetic_mean()
            elif choice == '4':
                self.weighted_mean()
            elif choice == '5':
                self.calculate_median()
            elif choice == '6':
                self.calculate_percentile()
            elif choice == '7':
                self.weighted_median()
            elif choice == '100':
                print("Exiting the program...")
                break
            else:
                print("Invalid choice. Please try again.")

# Statistics sınıfını kullanarak işlemleri gerçekleştirme
s = Statistics()
s.perform_operations()
