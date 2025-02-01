import numpy as np
import matplotlib.pyplot as plt

# Makalede verilen yöntem: Hibrit GA-TS + Zeckendorf

# Elektrik ve su talebi
electricity_demand = np.array([
    101000, 120000, 116000, 137000, 128000, 143000, 152500, 182000, 204000, 176000, 182000, 198000,
    169000, 168000, 169000, 157000, 157000, 157000, 172000, 143000, 185000, 227000, 229000, 213000,
    218000, 219000, 227000, 221000, 200500, 199500, 218000, 220000, 168000, 199500, 206000, 200000,
    193000, 169000, 181000, 177000, 183000, 163000, 133000, 108000, 123000, 112000, 120000, 98000,
    107000, 97000, 69000, 75000
])  # MW

water_demand = np.array([
    405, 405, 405, 405, 430, 430, 430, 430, 430, 430, 430, 430,
    430, 380, 380, 380, 380, 380, 485, 485, 485, 485, 510, 510,
    510, 510, 535, 535, 535, 535, 555, 555, 555, 555, 570, 570,
    570, 570, 570, 510, 510, 510, 510, 510, 390, 390, 390, 390,
    360, 360, 360, 360
])  # MIGD

# Ekipman bilgisi ve üretim kapasiteleri
equipment = {
    "boilers": {"count": 8, "duration": 5, "capacity": 0},
    "turbines": {
        "count": 8,
        "duration": 4,
        "capacity": np.array([47040] * 6 + [47040] * 2)  # MW
    },
    "distillers": {
        "count": 16,
        "duration": 5,
        "capacity": np.array([50.4] * 12 + [40.2] * 4)  # MIGD
    }
}

# Fibonacci serisi oluşturma
def generate_fibonacci_series(limit):
    series = [1, 2]
    while series[-1] + series[-2] <= limit:
        series.append(series[-1] + series[-2])
    return series

fibonacci_series = generate_fibonacci_series(52)

# Zeckendorf gösterimi
def zeckendorf_representation(number, fib_series):
    representation = []
    for fib in reversed(fib_series):
        if fib <= number:
            representation.append(1)
            number -= fib
        else:
            representation.append(0)
    return representation[::-1]

# Zeckendorf dönüşümünden sayı üretme
def zeckendorf_to_number(representation, fib_series):
    return sum(fib for bit, fib in zip(representation, fib_series) if bit == 1)

# Popülasyon oluşturma
def create_population(pop_size):
    population = []
    for _ in range(pop_size):
        schedule = {
            "electricity": np.ones((equipment["turbines"]["count"], 52)),  # **Başlangıçta tüm birimler çalışıyor (1)**
            "water": np.ones((equipment["distillers"]["count"], 52))  # **Başlangıçta tüm birimler çalışıyor (1)**
        }
        for eq_type in ["turbines", "distillers"]:
            for eq in range(equipment[eq_type]["count"]):
                start_week = zeckendorf_to_number(
                    zeckendorf_representation(np.random.randint(1, 52 - equipment[eq_type]["duration"]), fibonacci_series),
                    fibonacci_series
                )
                for w in range(start_week, min(start_week + equipment[eq_type]["duration"], 52)):
                    if eq_type == "turbines":
                        schedule["electricity"][eq, w] = 0  # **Bakım döneminde 0 atanıyor**
                    else:
                        schedule["water"][eq, w] = 0  # **Bakım döneminde 0 atanıyor**
        population.append(schedule)
    return population

# Fitness fonksiyonu
def fitness_function(schedule):
    electricity_production = np.sum(schedule["electricity"] * equipment["turbines"]["capacity"][..., np.newaxis], axis=0)
    water_production = np.sum(schedule["water"] * equipment["distillers"]["capacity"][..., np.newaxis], axis=0)

    electricity_gap =  electricity_production-electricity_demand
    water_gap = water_production-(water_demand)

    return np.sum(electricity_gap) + np.sum(water_gap)

# Genetik Algoritma
def genetic_algorithm(pop_size, generations, mutation_rate):
    population = create_population(pop_size)
    best_fitness = float("inf")
    best_schedule = None

    for gen in range(generations):
        fitness_values = [fitness_function(schedule) for schedule in population]
        best_idx = np.argmin(fitness_values)

        if fitness_values[best_idx] < best_fitness:
            best_fitness = fitness_values[best_idx]
            best_schedule = population[best_idx]

        print(f"Generation {gen + 1}: Best Fitness = {best_fitness}")

    return best_schedule, best_fitness

# Çalıştırma
best_schedule, best_fitness = genetic_algorithm(pop_size=50, generations=100, mutation_rate=0.1)

# Sonuçların görselleştirilmesi
electricity_production = np.sum(best_schedule["electricity"] * equipment["turbines"]["capacity"][..., np.newaxis], axis=0)
water_production = np.sum(best_schedule["water"] * equipment["distillers"]["capacity"][..., np.newaxis], axis=0)


total_electricity_production = np.sum(electricity_production)  # MW
total_water_production = np.sum(water_production)  # MIGD

# **Artırılan (fazla üretilen) elektrik ve su miktarları**
extra_electricity = electricity_production - electricity_demand
extra_water = water_production - water_demand


# Elektrik Grafiği
plt.figure(figsize=(12, 6))
plt.plot(electricity_demand, label="Electricity Demand", linestyle="--", color='r')
plt.plot(electricity_production, label="Electricity Production", color='b')
plt.plot(extra_electricity, label="Surplus", linestyle=":", color='g')
plt.title("Electricity Production and Demand")
plt.xlabel("Weeks")
plt.ylabel("MW")
plt.legend()
plt.show()

# Su Grafiği
plt.figure(figsize=(12, 6))
plt.plot(water_demand, label="Water Demand", linestyle="--", color='r')
plt.plot(water_production, label="Water Production", color='b')
plt.plot(extra_water, label="Surplus", linestyle=":", color='g')
plt.title("Water Production And Demand")
plt.xlabel("Weeks")
plt.ylabel("MIGD")
plt.legend()
plt.show()

print(f"Toplam Fitness: {best_fitness}")


# **Yıllık toplam üretim ve talep hesaplamaları**
total_electricity_production = np.sum(equipment["turbines"]["capacity"]) * 52
total_water_production = np.sum(equipment["distillers"]["capacity"]) * 52
total_electricity_demand = np.sum(electricity_demand)
total_water_demand = np.sum(water_demand)

def print_summary_table():
    electricity_difference = total_electricity_production - total_electricity_demand
    water_difference = total_water_production - total_water_demand

    electricity_diff_percentage = (electricity_difference / total_electricity_demand) * 100
    water_diff_percentage = (water_difference / total_water_demand) * 100

    print("\nYıllık Üretim ve Talep Karşılaştırması")
    print("=" * 50)
    print(f"{'Değer':<30}{'Üretim':<15}{'Talep':<15}{'Fark':<15}{'% Fark':<15}")
    print("-" * 50)
    print(f"{'Elektrik (MW)':<30}{total_electricity_production:<15.2f}{total_electricity_demand:<15.2f}{electricity_difference:<15.2f}{electricity_diff_percentage:<15.2f}")
    print(f"{'Su (MIGD)':<30}{total_water_production:<15.2f}{total_water_demand:<15.2f}{water_difference:<15.2f}{water_diff_percentage:<15.2f}")
    print("=" * 50)

print_summary_table()
