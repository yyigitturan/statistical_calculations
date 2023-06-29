import pandas as pd

class Stat:
    def __init__(self):
        self.data = []

    # Diğer fonksiyonlar buraya eklenecek...
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
    

    def generate_frequency_table(self, sinif_araligi, sinif_sayisi):
        if not self.data:
            print("No data available. Please enter data first.")
            return

        min_deger = min(self.data)
        max_deger = max(self.data)
        sinirler = [min_deger]
        for i in range(1, sinif_sayisi+1):
            sinirler.append(min_deger + i * sinif_araligi)

        sinif_ortaları = [(sinirler[i] + sinirler[i+1]) / 2 for i in range(len(sinirler)-1)]

        sinif_sikliklari = [0] * sinif_sayisi
        cumulative_frekans = [0] * sinif_sayisi
        toplam_siklik = 0
        for veri in self.data:
            for i in range(sinif_sayisi):
                if veri >= sinirler[i] and veri < sinirler[i+1]:
                    sinif_sikliklari[i] += 1
                    toplam_siklik += 1
                    cumulative_frekans[i] = toplam_siklik
                    break

        sinif_yuzdeleri = [(fi / toplam_siklik) * 100 for fi in sinif_sikliklari]

        tablo = pd.DataFrame({
            'Alt Sınır': sinirler[:-1],
            'Üst Sınır': sinirler[1:],
            'Sınıf Ortaları': sinif_ortaları,
            'fi': sinif_sikliklari,
            'Fi': cumulative_frekans,
            'pi': sinif_yuzdeleri
        })

        return tablo

    def arithmetic_mean(self):
        if self.data == None:
            print("data yok")
        else:
            return sum(self.data) / len(self.data)

# Kullanım örneği

stat = Stat()
stat.get_data_from_csv()  # CSV dosyasından veri almak için
frequency_table = stat.generate_frequency_table(5, 5)
print(frequency_table)
print(stat.arithmetic_mean())
