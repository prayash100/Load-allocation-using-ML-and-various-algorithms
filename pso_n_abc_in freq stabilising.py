import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define grid parameters
renewable_capacity = {'solar': 100, 'wind': 150}  # MW
conventional_capacity = {'coal': 200, 'gas': 100}  # MW

# Define the fluctuating load pattern with inertia
def fluctuating_load_pattern(base_demand, num_steps=100, max_variation=0.1, inertia=0.05):
    load = np.zeros(num_steps)
    load[0] = base_demand  # Starting load demand is the base demand
    
    for t in range(1, num_steps):
        fluctuation = np.random.uniform(-max_variation, max_variation) * base_demand
        load[t] = load[t-1] * (1 - inertia) + (base_demand + fluctuation) * inertia
        load[t] = max(0, load[t])  # Prevent negative load
        load[t] = min(base_demand * 1.5, load[t])  # Prevent excessively high load
    
    return load

base_demand = 500
load_demand = fluctuating_load_pattern(base_demand)

# Step 2: Renewable generation function
def renewable_generation(renewable_capacity):
    solar_output = renewable_capacity['solar'] * (np.random.rand() * 0.8 + 0.2)  
    wind_output = renewable_capacity['wind'] * (np.random.rand() * 0.7 + 0.3)   
    return solar_output, wind_output

# Step 3: Conventional generation function
def conventional_generation(remaining_demand, conventional_capacity):
    coal_output = min(conventional_capacity['coal'], remaining_demand)
    remaining_demand -= coal_output
    gas_output = min(conventional_capacity['gas'], remaining_demand)
    return coal_output, gas_output

# Step 4: Frequency deviation function
def grid_frequency(solar_output, wind_output, coal_output, gas_output, load_demand):
    total_generation = solar_output + wind_output + coal_output + gas_output
    frequency_deviation = 50 + (total_generation - load_demand) * 0.01  
    return frequency_deviation

# Step 5: Bee Search Algorithm (BSA) function
def bee_search_algorithm(renewable_capacity, conventional_capacity, load_demand, iterations):
    best_solution = np.random.rand(4) * [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']]
    best_deviation = np.inf
    
    for _ in range(iterations):
        new_solution = best_solution + (np.random.rand(4) - 0.5) * best_solution * 0.05
        new_solution = np.clip(new_solution, 0, [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']])
        
        solar_output, wind_output, coal_output, gas_output = new_solution
        frequency_deviation = grid_frequency(solar_output, wind_output, coal_output, gas_output, load_demand)
        
        if np.all(np.abs(frequency_deviation - 50) < best_deviation):  
            best_solution = new_solution
            best_deviation = np.abs(frequency_deviation - 50)
    
    return best_solution  

def particle_swarm_optimization(renewable_capacity, conventional_capacity, load_demand, iterations, swarm_size):
    swarm = np.random.rand(swarm_size, 4) * [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']]
    velocities = np.zeros_like(swarm)
    best_positions = np.copy(swarm)
    best_deviation = np.inf * np.ones(swarm_size)
    
    global_best_position = best_positions[0]
    global_best_deviation = np.inf

    for _ in range(iterations):
        for i in range(swarm_size):
            solar_output, wind_output, coal_output, gas_output = swarm[i]
            frequency_deviation = grid_frequency(solar_output, wind_output, coal_output, gas_output, load_demand)
            
            # Ensure that frequency_deviation is a scalar
            if isinstance(frequency_deviation, np.ndarray):
                frequency_deviation = frequency_deviation[0]  # Take the first element if it's an array

            # Calculate the absolute deviation from ideal frequency (50 Hz)
            abs_deviation = abs(frequency_deviation - 50)

            # Update best deviation for each particle
            if abs_deviation < best_deviation[i]:
                best_positions[i] = swarm[i]
                best_deviation[i] = abs_deviation

            # Update global best solution
            if abs_deviation < global_best_deviation:
                global_best_position = swarm[i]
                global_best_deviation = abs_deviation
        
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


# Step 7: Calculate frequency deviation with fluctuating load
def calculate_frequency_deviation_with_fluctuation(load_demand, renewable_capacity, conventional_capacity, best_solution):
    solar_output, wind_output, coal_output, gas_output = best_solution
    frequency_deviation = [
        grid_frequency(
            renewable_generation(renewable_capacity)[0],
            renewable_generation(renewable_capacity)[1],
            conventional_generation(load - sum(renewable_generation(renewable_capacity)), conventional_capacity)[0],
            conventional_generation(load - sum(renewable_generation(renewable_capacity)), conventional_capacity)[1],
            load
        ) for load in load_demand
    ]
    return frequency_deviation

# Step 8: Main execution
print('Starting simulation...')

# Run BSA and PSO
best_solution_bsa = bee_search_algorithm(renewable_capacity, conventional_capacity, load_demand, 1000)
print('Best solution found by BSA:')
print(best_solution_bsa)

# Calculate frequency deviation with fluctuating load for BSA
frequency_deviation_bsa = calculate_frequency_deviation_with_fluctuation(load_demand, renewable_capacity, conventional_capacity, best_solution_bsa)

best_solution_pso = particle_swarm_optimization(renewable_capacity, conventional_capacity, load_demand, 1000, 50)
print('Best solution found by PSO:')
print(best_solution_pso)

# Calculate frequency deviation with fluctuating load for PSO
frequency_deviation_pso = calculate_frequency_deviation_with_fluctuation(load_demand, renewable_capacity, conventional_capacity, best_solution_pso)






# Calculate frequency deviation with fluctuating load
frequency_deviation_with_fluctuation = [
    grid_frequency(
        renewable_generation(renewable_capacity)[0],
        renewable_generation(renewable_capacity)[1],
        conventional_generation(load - sum(renewable_generation(renewable_capacity)), conventional_capacity)[0],
        conventional_generation(load - sum(renewable_generation(renewable_capacity)), conventional_capacity)[1],
        load
    ) for load in load_demand
]

# Plot frequency deviation (BSA and PSO)
plt.figure(figsize=(12, 6))
plt.plot(frequency_deviation_bsa, label='Frequency Deviation (BSA)', color='blue')
plt.plot(frequency_deviation_pso, label='Frequency Deviation (PSO)', color='green')
#plt.plot(frequency_deviation_with_fluctuation, label='Frequency Deviation (Fluctuating Load)', color='black')
plt.axhline(y=50, color='red', linestyle='--', label='Ideal Frequency (50 Hz)')
plt.title('Frequency Deviation with Fluctuating Load')
plt.xlabel('Time Steps')
plt.ylabel('Frequency (Hz)')
plt.legend()
plt.show()
