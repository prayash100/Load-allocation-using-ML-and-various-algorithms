import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
load_model = keras.models.load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import os
from dotenv import load_dotenv
load_dotenv()
import tensorflow as tf
tf.get_logger().setLevel("ERROR")  # Suppress TensorFlow logs 

import pandas as pd
# Step 1: Define grid parameters
renewable_capacity = {'solar':2000, 'wind': 4000}  # MW
conventional_capacity = {'coal': 12000, 'gas': 10000}  # MW

scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

df = pd.read_csv("corrected_gujarat_load_demand_2024.csv")  # Replace with your file path
df["Index"] = range(len(df))
df['Load_Demand_MW'] = df['Load_Demand_MW'].round(2)
df_w = df.iloc[20000:20200]  

load_demand = df_w['Load_Demand_MW'].values

def load_lstm_model(model_path='lstm_load_prediction.h5'):
    return load_model(model_path, compile=False)

def predict_next_100_seconds(last_200_seconds, model):
    
    
    # Reshape input
    last_200_seconds = np.array(last_200_seconds).reshape(1, -1)
    
    
    last_200_seconds_scaled = scaler_X.transform(last_200_seconds).reshape(1, 200, 1)
    

    # Predict
    predicted_scaled = model.predict(last_200_seconds_scaled)[0]
    

    # Inverse transform
    predicted_actual = scaler_y.inverse_transform(predicted_scaled.reshape(1, -1))[0]

    return predicted_actual

def renewable_generation(renewable_capacity):
    solar_output = renewable_capacity['solar'] * (np.random.rand() * 0.8 + 0.2)  # Random output between 20% to 100% of capacity
    wind_output = renewable_capacity['wind'] * (np.random.rand() * 0.7 + 0.3)   # Random output between 30% to 100% of capacity
    return solar_output, wind_output

def conventional_generation(remaining_demand, conventional_capacity):
    if isinstance(remaining_demand, np.ndarray):  # If remaining_demand is an array
        coal_output = np.minimum(0.8*conventional_capacity['coal'], remaining_demand)
        remaining_demand -= coal_output  # Reduce remaining demand by coal output
        gas_output = np.minimum(conventional_capacity['gas'], remaining_demand)
    else:  # If remaining_demand is a scalar
        coal_output = min(0.8*conventional_capacity['coal'], remaining_demand)
        remaining_demand -= coal_output
        gas_output = min(conventional_capacity['gas'], remaining_demand)

    return coal_output, gas_output

def grid_frequency(solar_output, wind_output, coal_output, gas_output, load_demand):
    total_generation = solar_output + wind_output + coal_output + gas_output 
    delta_P = total_generation - load_demand  # Power mismatch

    # Inertia constants (H), lower for RES, higher for coal/gas
    H_solar, H_wind, H_coal, H_gas = 0.05, 0.1, 5.0, 3.0
    D_solar, D_wind, D_coal, D_gas= 0.02, 0.03, 0.3, 0.2  # Damping factors

    # Weighted total inertia and damping effect
    H_total = (solar_output * H_solar + wind_output * H_wind +
               coal_output * H_coal + gas_output * H_gas  ) / (total_generation + 1e-6)

    D_total = (solar_output * D_solar + wind_output * D_wind +
               coal_output * D_coal + gas_output * D_gas  ) / (total_generation + 1e-6)

    # Frequency deviation is higher if RES contribution is higher
    frequency_deviation = (delta_P * 0.000011) / (H_total + D_total + 1e-6)  # Prevent div by zero
    return frequency_deviation

def bee_search_algorithm(renewable_capacity, conventional_capacity, data, iterations):
    best_solution = np.random.rand(4) * [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']]
    best_deviation = np.inf
    
    for _ in range(iterations):
        new_solution = best_solution + (np.random.rand(4) - 0.5) * best_solution * 0.05
        new_solution = np.clip(new_solution, 0, [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']])
        
        solar_output, wind_output, coal_output, gas_output = new_solution
        frequency_deviation = grid_frequency(solar_output, wind_output, coal_output, gas_output, data)
        
        if np.all(np.abs(frequency_deviation) < best_deviation):  # Use np.all() for array comparison
            best_solution = new_solution
            best_deviation = np.abs(frequency_deviation)
    
    return np.trunc(best_solution * 100) / 100  # Return the best solution found

def particle_swarm_optimization(renewable_capacity, conventional_capacity, data, iterations, swarm_size=50):
    swarm = np.random.rand(swarm_size, 4) * [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']]
    velocities = np.zeros_like(swarm)
    best_positions = np.copy(swarm)
    best_deviation = np.inf * np.ones(swarm_size)
    
    global_best_position = best_positions[0]
    global_best_deviation = np.inf

    for _ in range(iterations):
        for i in range(swarm_size):
            solar_output, wind_output, coal_output, gas_output = swarm[i]
            frequency_deviation = grid_frequency(solar_output, wind_output, coal_output, gas_output, data)
            
            # Ensure that frequency_deviation is a scalar value
            if isinstance(frequency_deviation, np.ndarray):
                frequency_deviation = frequency_deviation[0]  # Take the first element if it's an array

            # Update best deviation for each particle
            if abs(frequency_deviation) < best_deviation[i]:
                best_positions[i] = swarm[i]
                best_deviation[i] = abs(frequency_deviation)

            # Update global best solution
            if abs(frequency_deviation) < global_best_deviation:
                global_best_position = swarm[i]
                global_best_deviation = abs(frequency_deviation)
        
        inertia_weight = 0.9
        cognitive_weight = 1.5
        social_weight = 1.4
        
        for i in range(swarm_size):
            velocities[i] = inertia_weight * velocities[i] + \
                            cognitive_weight * np.random.rand() * (best_positions[i] - swarm[i]) + \
                            social_weight * np.random.rand() * (global_best_position - swarm[i])
            swarm[i] = swarm[i] + velocities[i]
            swarm[i] = np.clip(swarm[i], 0, [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']])
    
    return  np.trunc(global_best_position * 100) / 100

def genetic_algorithm(renewable_capacity, conventional_capacity, data, iterations):
    population_size = 10
    population = np.random.rand(population_size, 4) * [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']]
    best_solution = population[0]
    best_deviation = np.inf
    
    for _ in range(iterations):
        for i in range(population_size):
            solar_output, wind_output, coal_output, gas_output = population[i]
            frequency_deviation = grid_frequency(solar_output, wind_output, coal_output, gas_output, data)
            
            if np.all(np.abs(frequency_deviation) < best_deviation):
                best_solution = population[i]
                best_deviation = np.abs(frequency_deviation)
        
        new_population = population + (np.random.rand(population_size, 4) - 0.5) * 0.1
        population = np.clip(new_population, 0, [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']])
    
    return  np.trunc(best_solution * 100) / 100

def differential_evolution(renewable_capacity, conventional_capacity, data, iterations):
    population_size = 10
    population = np.random.rand(population_size, 4) * [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']]
    best_solution = population[0]
    best_deviation = np.inf
    
    for _ in range(iterations):
        for i in range(population_size):
            mutant = population[i] + 0.5 * (population[(i+1)%population_size] - population[(i+2)%population_size])
            mutant = np.clip(mutant, 0, [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']])
            frequency_deviation = grid_frequency(*mutant, data)
            
            if np.all(np.abs(frequency_deviation) < best_deviation):
                best_solution = mutant
                best_deviation = np.abs(frequency_deviation)
    
    return  np.trunc(best_solution * 100) / 100

def simulated_annealing(renewable_capacity, conventional_capacity, data, iterations):
    best_solution = np.random.rand(4) * [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']]
    best_deviation = np.inf
    temperature = 100
    cooling_rate = 0.95
    
    for _ in range(iterations):
        new_solution = best_solution + (np.random.rand(4) - 0.5) * best_solution * 0.1
        new_solution = np.clip(new_solution, 0, [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']])
        
        frequency_deviation = grid_frequency(*new_solution, data)
        deviation_magnitude = np.mean(np.abs(frequency_deviation))  # Ensure scalar comparison
        
        if deviation_magnitude < best_deviation or np.exp(-(deviation_magnitude - best_deviation) / temperature) > np.random.rand():
            best_solution = new_solution
            best_deviation = deviation_magnitude
        
        temperature *= cooling_rate
    
    return  np.trunc(best_solution * 100) / 100

def calculate_total_cost(solar_output, wind_output, coal_output, gas_output):
    # Adjusted costs to reflect market prices more realistically
    cost_per_mw = {
        'solar': 50,  # $/MWh
        'wind': 90,   # $/MWh
        'coal': 130,   # $/MWh
        'gas': 60     # $/MWh
    }
    
    # Ensure inputs are NumPy arrays
    solar_output = np.asarray(solar_output)
    wind_output = np.asarray(wind_output)
    coal_output = np.asarray(coal_output)
    gas_output = np.asarray(gas_output)
    
    # Calculate total cost
    total_cost = np.sum(
        solar_output * cost_per_mw['solar'] + 
        wind_output * cost_per_mw['wind'] + 
        coal_output * cost_per_mw['coal'] + 
        gas_output * cost_per_mw['gas']
    )
    
    return total_cost

def plot_frequency_stability(solution, load_demand, num_steps=100):
    # Extract individual power sources from the solution
    solar_output, wind_output, coal_output, gas_output = solution

    # Array to store frequency deviations over time
    frequency_obtained = []

    # Simulate frequency deviations over time
    for t in range(num_steps):
        # Get the total power generation at this time step
        total_generation = solar_output + wind_output + coal_output + gas_output
        delta_P = total_generation - load_demand[t]  # Power mismatch

        # Inertia and damping constants
        H_solar, H_wind, H_coal, H_gas = 0.05, 0.1, 5.0, 3.0
        D_solar, D_wind, D_coal, D_gas = 0.02, 0.03, 0.3, 0.2

        # Weighted total inertia and damping effect
        H_total = (solar_output * H_solar + wind_output * H_wind +
                   coal_output * H_coal + gas_output * H_gas) / (total_generation + 1e-6)

        D_total = (solar_output * D_solar + wind_output * D_wind +
                   coal_output * D_coal + gas_output * D_gas) / (total_generation + 1e-6)

        # Calculate frequency deviation
        frequency_deviation = (delta_P * 0.000011) / (H_total + D_total + 1e-6)  # Prevent div by zero
        obtained_hz = 50 + frequency_deviation  # Adjusted frequency

        frequency_obtained.append(obtained_hz)

    # Plot the frequency stability over time
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_steps), frequency_obtained, label="Grid Frequency (Hz)", color="blue")
    plt.axhline(y=50, color="red", linestyle="--", label="Nominal Frequency (50 Hz)")
    plt.xlabel("Time Steps")
    plt.ylabel("Frequency (Hz)")
    plt.title("Grid Frequency Stability Over Time")
    plt.legend()
    plt.grid()
    plt.show()

def frequency_array(solution, load_demand, num_steps=100):
    # Arrays to store frequency deviations over time
    frequency_obtained = []

    # Simulate frequency deviations over time
    for t in range(num_steps):
        total = np.sum(solution)
        deviation_mw = total - load_demand[t]
        # Calculate frequency deviation
        obtained_hz = 50+(deviation_mw * 0.000011)  # Adjust sensitivity as needed
        frequency_obtained.append(obtained_hz)
    return np.array(frequency_obtained) 

def generate_frequency_report(frequencies):
    
    ideal_freq = 50
    frequencies = np.array(frequencies)  # Ensure it's a NumPy array
    
    # Calculate deviations
    deviations = np.abs(frequencies - ideal_freq)
    max_deviation = np.max(deviations)
    min_deviation = np.min(deviations)
    mean_frequency = np.mean(frequencies)
    std_dev = np.std(frequencies)

    # Percentage of values within tolerance ranges
    within_0_01 = np.sum(deviations <= 0.01) / len(frequencies) * 100
    within_0_02 = np.sum(deviations <= 0.02) / len(frequencies) * 100
    within_0_05 = np.sum(deviations <= 0.05) / len(frequencies) * 100

    # Print Report
    print("="*40)
    print("       Frequency Stability Report")
    print("="*40)
    print(f"Total Readings: {len(frequencies)}")
    print(f"Mean Frequency: {mean_frequency:.3f} Hz")
    print(f"Standard Deviation: {std_dev:.3f} Hz")
    print(f"Maximum Deviation: {max_deviation:.3f} Hz")
    print(f"Minimum Deviation: {min_deviation:.3f} Hz")
    print("-"*40)
    print(f"Percentage of values within ±0.01 Hz: {within_0_01:.2f}%")
    print(f"Percentage of values within ±0.02 Hz: {within_0_02:.2f}%")
    print(f"Percentage of values within ±0.05 Hz: {within_0_05:.2f}%")
    print("="*40)
    
data=predict_next_100_seconds(load_demand, load_lstm_model())


# Main execution
print("="*60)
print("="*60)
print('          Starting simulation...')
print("="*60)
print("="*60)

# Plot the fluctuating load demand 200 seconds
plt.plot(load_demand)
plt.xlabel('Time Step')
plt.ylabel('Load Demand (MW)')
plt.title('Fluctuating Load Demand with Inertia')
plt.grid(True)
plt.show() 

# Plot the fluctuating load demand next 100 seconds
plt.plot(data)
plt.xlabel('Time Step')
plt.ylabel('Load Demand (MW)')
plt.title('Fluctuating Load Demand with Inertia')
plt.grid(True)
plt.show()

# Run BSA 
best_solution_bsa = bee_search_algorithm(renewable_capacity, conventional_capacity, data, 1000)
print("="*40)
print('        BSA')
print("="*40)
print('[SOLAR    WIND   COAL     GAS]')
print(best_solution_bsa)
solar_output_bsa, wind_output_bsa, coal_output_bsa, gas_output_bsa = best_solution_bsa
total_cost_bsa = calculate_total_cost(solar_output_bsa, wind_output_bsa, coal_output_bsa, gas_output_bsa)
print(f'Total cost after BSA: {total_cost_bsa:.2f}')
plot_frequency_stability(best_solution_bsa, data)
a=frequency_array(best_solution_bsa, data)
generate_frequency_report(a)

#Run PSO
best_solution_pso = particle_swarm_optimization(renewable_capacity, conventional_capacity, data, 1000)
print("="*40)
print('        PSO:')
print("="*40)
print('[SOLAR    WIND   COAL     GAS]')
print(best_solution_pso)
solar_output_pso, wind_output_pso, coal_output_pso, gas_output_pso = best_solution_pso
total_cost_pso = calculate_total_cost(solar_output_pso, wind_output_pso, coal_output_pso, gas_output_pso)
print(f'Total cost after BSA: {total_cost_pso:.2f}')
plot_frequency_stability(best_solution_pso, data)
a=frequency_array(best_solution_pso, data)
generate_frequency_report(a)

#Run GA
best_solution_ga = genetic_algorithm(renewable_capacity, conventional_capacity, data, 1000)
print("="*40)
print('        GA:')
print("="*40)
print('[SOLAR    WIND   COAL     GAS]')
print(best_solution_ga)
solar_output_ga, wind_output_ga, coal_output_ga, gas_output_ga = best_solution_ga
total_cost_ga = calculate_total_cost(solar_output_ga, wind_output_ga, coal_output_ga, gas_output_ga)
print(f'Total cost after GA: {total_cost_ga:.2f}')
plot_frequency_stability(best_solution_ga, data)
a=frequency_array(best_solution_ga, data)
generate_frequency_report(a)

#Run DE
best_solution_de = differential_evolution(renewable_capacity, conventional_capacity, data, 1000)
print("="*40)
print('        DE:')
print("="*40)
print('[SOLAR    WIND   COAL     GAS]')
print(best_solution_de)
solar_output_de, wind_output_de, coal_output_de, gas_output_de = best_solution_de
total_cost_de = calculate_total_cost(solar_output_de, wind_output_de, coal_output_de, gas_output_de)
print(f'Total cost after DE: {total_cost_de:.2f}')
plot_frequency_stability(best_solution_de, data)
a=frequency_array(best_solution_de, data)
generate_frequency_report(a)

#Run SA
best_solution_sa = simulated_annealing(renewable_capacity, conventional_capacity, data, 1000)
print("="*40)
print('        SA:')
print("="*40)
print('[SOLAR    WIND   COAL     GAS]')
print(best_solution_sa)
solar_output_sa, wind_output_sa, coal_output_sa, gas_output_sa = best_solution_sa
total_cost_sa = calculate_total_cost(solar_output_sa, wind_output_sa, coal_output_sa, gas_output_sa)
print(f'Total cost after SA: {total_cost_sa:.2f}')
plot_frequency_stability(best_solution_sa, data)
a=frequency_array(best_solution_sa, data)
generate_frequency_report(a)

 
print("="*60)
print("="*60)
print('          Simulation completed.')
print("="*60)
print("="*60)







