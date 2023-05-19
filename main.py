import pandas as pd

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
            num = input("Enter a number (to quit, enter 'q'): ")
            if num.lower() == 'q':
                break
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

    def perform_operations(self):
        while True:
            print("\nOperations Menu:")
            print("1. Enter data manually")
            print("2. Import data from CSV")
            print("-------------------------")
            print("3. Calculate arithmetic mean")
            print("4. Calculate weighted mean")
            print("5. Exit")

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
                print("Exiting the program...")
                break
            else:
                print("Invalid choice. Please try again.")

# Statistics sınıfını kullanarak işlemleri gerçekleştirme
s = Statistics()
s.perform_operations()
