import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
load_model = keras.models.load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import os
from dotenv import load_dotenv
load_dotenv()
import tensorflow as tf
tf.get_logger().setLevel("ERROR")  # Suppress TensorFlow logs 

# Step 1: Define grid parameters
renewable_capacity = {'solar':2000, 'wind': 4000}  # MW
conventional_capacity = {'coal': 12000, 'gas': 10000}  # MW


scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Define the fluctuating load pattern with inertia
# Function to generate 200 seconds of fluctuating load demand
def fluctuating_load_pattern(base_demand, num_steps=200, max_variation=0.2, inertia=0.05):
    """
    Generates a fluctuating load demand pattern with inertia.
    base_demand: The base load demand around which fluctuations occur.
    num_steps: The number of time steps for the simulation.
    max_variation: The maximum variation (percentage) in load demand.
    inertia: Controls how much the previous value affects the current value.
    """
    load = np.zeros(num_steps)
    load[0] = base_demand  # Starting load demand is the base demand
    
    for t in range(1, num_steps):
        # Generate random fluctuation
        fluctuation = np.random.uniform(-max_variation, max_variation) * base_demand
        # Apply inertia (smooth out large fluctuations)
        load[t] = load[t-1] * (1 - inertia) + (base_demand + fluctuation) * inertia
        
        # Ensure that the load demand stays within reasonable bounds
        load[t] = max(0, load[t])  # Prevent negative load
        load[t] = min(base_demand * 1.5, load[t])  # Prevent excessively high load
    
    return load
base_demand = 25000  # 25,000 MW
load_demand = fluctuating_load_pattern(base_demand)
def load_lstm_model(model_path='lstm_load_prediction.h5'):
    print(f"Loading LSTM model from {model_path}...")
    return load_model(model_path, compile=False)

def predict_next_100_seconds(last_200_seconds, model):
    print("\n[INFO] Predicting next 100 seconds of load...")
    
    # Reshape input
    last_200_seconds = np.array(last_200_seconds).reshape(1, -1)
    print(f"[DEBUG] Shape before scaling: {last_200_seconds.shape}")
    
    last_200_seconds_scaled = scaler_X.transform(last_200_seconds).reshape(1, 200, 1)
    print(f"[DEBUG] Shape after scaling: {last_200_seconds_scaled.shape}")

    # Predict
    predicted_scaled = model.predict(last_200_seconds_scaled)[0]
    print(f"[DEBUG] Raw scaled prediction shape: {predicted_scaled.shape}")

    # Inverse transform
    predicted_actual = scaler_y.inverse_transform(predicted_scaled.reshape(1, -1))[0]
    print(f"[DEBUG] Final prediction shape: {predicted_actual.shape}")

    return predicted_actual

data=predict_next_100_seconds(load_demand, load_lstm_model())



# Step 2: Renewable generation function
def renewable_generation(renewable_capacity):
    solar_output = renewable_capacity['solar'] * (np.random.rand() * 0.8 + 0.2)  # Random output between 20% to 100% of capacity
    wind_output = renewable_capacity['wind'] * (np.random.rand() * 0.7 + 0.3)   # Random output between 30% to 100% of capacity
    return solar_output, wind_output

# Step 3: Conventional generation function
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

# Step 5: Bee Search Algorithm (BSA) function
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
    
    return best_solution  # Return the best solution found


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
    
    return global_best_position

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
    
    return best_solution

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
    
    return best_solution

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
    
    return best_solution




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

# Step 8: Main execution
print('Starting simulation...')

# Run BSA 
best_solution_bsa = bee_search_algorithm(renewable_capacity, conventional_capacity, data, 1000)
print('Best solution found by BSA:')
print('[SOLAR    WIND   COAL     GAS]')
print(best_solution_bsa)
solar_output_bsa, wind_output_bsa, coal_output_bsa, gas_output_bsa = best_solution_bsa
total_cost_bsa = calculate_total_cost(solar_output_bsa, wind_output_bsa, coal_output_bsa, gas_output_bsa)
print(f'Total cost after BSA: {total_cost_bsa:.2f}')


#Run PSO
best_solution_pso = particle_swarm_optimization(renewable_capacity, conventional_capacity, data, 1000)
print('Best solution found by PSO:')
print('[SOLAR    WIND   COAL     GAS]')
print(best_solution_pso)
solar_output_pso, wind_output_pso, coal_output_pso, gas_output_pso = best_solution_pso
total_cost_pso = calculate_total_cost(solar_output_pso, wind_output_pso, coal_output_pso, gas_output_pso)
print(f'Total cost after BSA: {total_cost_pso:.2f}')


#Run GA
best_solution_ga = genetic_algorithm(renewable_capacity, conventional_capacity, data, 1000)
print('Best solution found by GA:')
print('[SOLAR    WIND   COAL     GAS]')
print(best_solution_ga)
solar_output_ga, wind_output_ga, coal_output_ga, gas_output_ga = best_solution_ga
total_cost_ga = calculate_total_cost(solar_output_ga, wind_output_ga, coal_output_ga, gas_output_ga)
print(f'Total cost after GA: {total_cost_ga:.2f}')

#Run DE
best_solution_de = differential_evolution(renewable_capacity, conventional_capacity, data, 1000)
print('Best solution found by DE:')
print('[SOLAR    WIND   COAL     GAS]')
print(best_solution_de)
solar_output_de, wind_output_de, coal_output_de, gas_output_de = best_solution_de
total_cost_de = calculate_total_cost(solar_output_de, wind_output_de, coal_output_de, gas_output_de)
print(f'Total cost after DE: {total_cost_de:.2f}')

#Run SA
best_solution_sa = simulated_annealing(renewable_capacity, conventional_capacity, data, 1000)
print('Best solution found by SA:')
print('[SOLAR    WIND   COAL     GAS]')
print(best_solution_sa)
solar_output_sa, wind_output_sa, coal_output_sa, gas_output_sa = best_solution_sa
total_cost_sa = calculate_total_cost(solar_output_sa, wind_output_sa, coal_output_sa, gas_output_sa)
print(f'Total cost after SA: {total_cost_sa:.2f}')

# Plot the fluctuating load demand
plt.plot(load_demand)
plt.xlabel('Time Step')
plt.ylabel('Load Demand (MW)')
plt.title('Fluctuating Load Demand with Inertia')
plt.grid(True)
plt.show() 

# Plot the fluctuating load demand
plt.plot(data)
plt.xlabel('Time Step')
plt.ylabel('Load Demand (MW)')
plt.title('Fluctuating Load Demand with Inertia')
plt.grid(True)
plt.show() 

# Step 9: Frequency Deviation Plot Function with 50 Hz Base
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


# Call the function to plot frequency stability
plot_frequency_stability(best_solution_bsa, data)
plot_frequency_stability(best_solution_pso, data)
plot_frequency_stability(best_solution_ga, data)
plot_frequency_stability(best_solution_de, data)
plot_frequency_stability(best_solution_sa, data)


