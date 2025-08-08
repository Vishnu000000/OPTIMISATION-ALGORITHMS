import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import random

# --- Part 1: Objective Functions for Demonstration ---
def rastrigin_function(x):
    """A classic multi-modal function with many local minima."""
    x = np.asarray(x)
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def sphere_function(x):
    """A simple unimodal function for minimization."""
    x = np.asarray(x)
    return np.sum(x**2)

# --- Part 2: Integrated Optimization Algorithms ---

# The following functions are simplified adaptations of your uploaded code files
# to work within the Streamlit app's visualization framework.

def run_genetic_algorithm(objective_func, bounds, num_iterations, pop_size, mutation_rate):
    num_dimensions = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    population = np.random.uniform(lower_bounds, upper_bounds, size=(pop_size, num_dimensions))
    best_solution = None
    best_fitness = np.inf
    history = []

    for _ in range(num_iterations):
        fitness = np.array([objective_func(p) for p in population])
        if np.min(fitness) < best_fitness:
            best_fitness = np.min(fitness)
            best_solution = population[np.argmin(fitness)]
        history.append(population.copy())

        # Simple selection (roulette wheel), crossover, and mutation logic
        fitness_inv = 1 / (fitness + 1e-6) # Invert for minimization
        selection_probabilities = fitness_inv / np.sum(fitness_inv)
        parent_indices = np.random.choice(pop_size, size=pop_size, p=selection_probabilities)
        parents = population[parent_indices]

        children = []
        for j in range(0, pop_size, 2):
            if j + 1 < pop_size:
                crossover_point = np.random.randint(1, num_dimensions)
                child1 = np.concatenate([parents[j][:crossover_point], parents[j+1][crossover_point:]])
                child2 = np.concatenate([parents[j+1][:crossover_point], parents[j][:crossover_point:]])
                children.append(child1)
                children.append(child2)

        for child in children:
            if np.random.rand() < mutation_rate:
                mutation_idx = np.random.randint(num_dimensions)
                child[mutation_idx] = np.random.uniform(lower_bounds[mutation_idx], upper_bounds[mutation_idx])
            np.clip(child, lower_bounds, upper_bounds, out=child)
        
        population = np.array(children)
    
    return best_solution, history, best_fitness

def run_pso(objective_func, bounds, num_iterations, num_particles, w, c1_pso, c2_pso):
    num_dimensions = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    particles = np.random.uniform(lower_bounds, upper_bounds, size=(num_particles, num_dimensions))
    velocities = np.zeros_like(particles)
    pbest_positions = particles.copy()
    pbest_values = np.full(num_particles, np.inf)
    gbest_position = None
    gbest_value = np.inf
    history = []

    for _ in range(num_iterations):
        history.append(particles.copy())
        for i in range(num_particles):
            objective_value = objective_func(particles[i])
            if objective_value < pbest_values[i]:
                pbest_values[i] = objective_value
                pbest_positions[i] = particles[i].copy()
            if objective_value < gbest_value:
                gbest_value = objective_value
                gbest_position = particles[i].copy()

        for i in range(num_particles):
            r1 = np.random.rand(num_dimensions)
            r2 = np.random.rand(num_dimensions)
            velocities[i] = (w * velocities[i] +
                             c1_pso * r1 * (pbest_positions[i] - particles[i]) +
                             c2_pso * r2 * (gbest_position - particles[i]))
            particles[i] += velocities[i]
            np.clip(particles[i], lower_bounds, upper_bounds, out=particles[i])

    return gbest_position, history, gbest_value

def run_sa(objective_func, bounds, num_iterations, T_max, T_min, alpha):
    num_dimensions = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    current_solution = np.random.uniform(lower_bounds, upper_bounds)
    current_optimal = objective_func(current_solution)
    best_solution = current_solution
    best_optimal = current_optimal
    history = [current_solution.copy()]

    for i in range(num_iterations):
        T = T_max * (T_min / T_max)**(i / num_iterations)
        new_solution = current_solution + np.random.normal(0, alpha * (upper_bounds - lower_bounds))
        new_solution = np.clip(new_solution, lower_bounds, upper_bounds)
        new_optimal = objective_func(new_solution)

        if new_optimal < current_optimal or np.random.rand() < np.exp((current_optimal - new_optimal) / T):
            current_solution = new_solution
            current_optimal = new_optimal
        
        if current_optimal < best_optimal:
            best_solution = current_solution
            best_optimal = current_optimal
        
        history.append(current_solution.copy())
    
    # SA returns a list of individual solutions over time, not a population.
    # We will treat this list as a 'population' for plotting purposes.
    return best_solution, history, best_optimal

def run_tlbo(objective_func, bounds, num_iterations, pop_size):
    num_dimensions = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    population = np.random.uniform(lower_bounds, upper_bounds, size=(pop_size, num_dimensions))
    history = []
    best_solution = None
    best_fitness = np.inf
    
    for i in range(num_iterations):
        history.append(population.copy())
        objectives = np.array([objective_func(x) for x in population])
        
        if np.min(objectives) < best_fitness:
            best_fitness = np.min(objectives)
            best_solution = population[np.argmin(objectives)]

        # Teacher Phase
        best_index = np.argmin(objectives)
        teacher_solution = population[best_index]
        mean_solution = np.mean(population, axis=0)
        
        new_population = []
        for x in population:
            rand_factor = np.random.rand(num_dimensions)
            new_x = x + rand_factor * (teacher_solution - mean_solution)
            new_x = np.clip(new_x, lower_bounds, upper_bounds)
            if objective_func(new_x) < objective_func(x):
                new_population.append(new_x)
            else:
                new_population.append(x)
        population = np.array(new_population)

        # Learner Phase
        new_population = []
        for i in range(pop_size):
            x_i = population[i]
            j = random.randint(0, pop_size - 1)
            x_j = population[j]
            rand_factor = np.random.rand(num_dimensions)

            if objective_func(x_i) < objective_func(x_j):
                new_x = x_i + rand_factor * (x_i - x_j)
            else:
                new_x = x_i + rand_factor * (x_j - x_i)

            new_x = np.clip(new_x, lower_bounds, upper_bounds)
            if objective_func(new_x) < objective_func(x_i):
                new_population.append(new_x)
            else:
                new_population.append(x_i)

        population = np.array(new_population)

    return best_solution, history, best_fitness

def run_abc(objective_func, bounds, num_iterations, num_employed_bees):
    # This is a simplified ABC algorithm focusing on the core concepts.
    num_dimensions = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    pop_size = num_employed_bees * 2
    population = np.random.uniform(lower_bounds, upper_bounds, size=(pop_size, num_dimensions))
    fitness = np.array([objective_func(x) for x in population])
    
    history = []
    best_solution = None
    best_fitness = np.inf
    
    for _ in range(num_iterations):
        history.append(population.copy())
        
        # Employed Bees
        for i in range(num_employed_bees):
            k = np.random.randint(num_employed_bees)
            while i == k:
                k = np.random.randint(num_employed_bees)
            
            j = np.random.randint(num_dimensions)
            phi = np.random.uniform(-1, 1)
            
            new_solution = population[i].copy()
            new_solution[j] += phi * (population[i, j] - population[k, j])
            new_solution = np.clip(new_solution, lower_bounds, upper_bounds)
            
            new_fitness = objective_func(new_solution)
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        # Onlooker Bees
        # Fix: Using inverse fitness for minimization probability calculation.
        fitness_inv = 1 / (fitness + 1e-6)
        probs = fitness_inv[:num_employed_bees] / np.sum(fitness_inv[:num_employed_bees])
        
        for i in range(num_employed_bees):
            idx = np.random.choice(num_employed_bees, p=probs)
            k = np.random.randint(num_employed_bees)
            while idx == k:
                k = np.random.randint(num_employed_bees)
            
            j = np.random.randint(num_dimensions)
            phi = np.random.uniform(-1, 1)
            
            new_solution = population[idx].copy()
            new_solution[j] += phi * (population[idx, j] - population[k, j])
            new_solution = np.clip(new_solution, lower_bounds, upper_bounds)
            
            new_fitness = objective_func(new_solution)
            if new_fitness < fitness[idx]:
                population[idx] = new_solution
                fitness[idx] = new_fitness

        if np.min(fitness) < best_fitness:
            best_fitness = np.min(fitness)
            best_solution = population[np.argmin(fitness)]
            
    return best_solution, history, best_fitness

def run_csa(objective_func, bounds, num_iterations, num_cuckoos, pa):
    num_dimensions = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    nests = np.random.uniform(lower_bounds, upper_bounds, size=(num_cuckoos, num_dimensions))
    fitness = np.array([objective_func(x) for x in nests])
    history = []
    
    for _ in range(num_iterations):
        history.append(nests.copy())
        
        # Get a cuckoo
        cuckoo_idx = np.random.randint(num_cuckoos)
        cuckoo = nests[cuckoo_idx]
        
        # Levy flight to get new position
        lambda_val = 1.5
        sigma_num = (np.random.gamma(1 + lambda_val) * np.sin(np.pi * lambda_val / 2))
        sigma_den = (np.random.gamma((1 + lambda_val) / 2) * lambda_val * (2**((lambda_val - 1) / 2)))
        sigma = sigma_num / sigma_den
        u = np.random.normal(0, sigma, size=num_dimensions)
        v = np.random.normal(0, 1, size=num_dimensions)
        step = u / (np.abs(v)**(1 / lambda_val))
        step_size = 0.01 * step * (cuckoo - nests[np.argmin(fitness)])
        new_cuckoo = cuckoo + step_size
        
        # Check if new cuckoo is better
        if objective_func(new_cuckoo) < fitness[cuckoo_idx]:
            nests[cuckoo_idx] = new_cuckoo
            fitness[cuckoo_idx] = objective_func(new_cuckoo)
            
        # Abandon some nests
        abandon_nests = np.random.rand(num_cuckoos) < pa
        for i in range(num_cuckoos):
            if abandon_nests[i]:
                nests[i] = np.random.uniform(lower_bounds, upper_bounds)
                fitness[i] = objective_func(nests[i])

    best_fitness = np.min(fitness)
    best_solution = nests[np.argmin(fitness)]
    return best_solution, history, best_fitness

def run_alo(objective_func, bounds, num_iterations, num_agents):
    num_dimensions = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    antlion_positions = np.random.uniform(lower_bounds, upper_bounds, size=(num_agents, num_dimensions))
    best_solution = None
    best_optimal = np.inf
    history = []
    
    for _ in range(num_iterations):
        history.append(antlion_positions.copy())
        objective_values = np.array([objective_func(pos) for pos in antlion_positions])
        
        if np.min(objective_values) < best_optimal:
            best_optimal = np.min(objective_values)
            best_solution = antlion_positions[np.argmin(objective_values)]

        # Update antlion positions
        for i in range(num_agents):
            distance_to_best = np.linalg.norm(best_solution - antlion_positions[i])
            scale_factor = np.exp(-distance_to_best)
            antlion_positions[i] += scale_factor * np.random.uniform(-1, 1, size=num_dimensions)
            antlion_positions[i] = np.clip(antlion_positions[i], lower_bounds, upper_bounds)

    return best_solution, history, best_optimal

def run_dragonfly(objective_func, bounds, num_iterations, num_dragonflies):
    num_dimensions = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    dragonfly_positions = np.random.uniform(lower_bounds, upper_bounds, size=(num_dragonflies, num_dimensions))
    best_solution = None
    best_optimal = np.inf
    history = []
    
    for _ in range(num_iterations):
        history.append(dragonfly_positions.copy())
        objective_values = np.array([objective_func(pos) for pos in dragonfly_positions])

        if np.min(objective_values) < best_optimal:
            best_optimal = np.min(objective_values)
            best_solution = dragonfly_positions[np.argmin(objective_values)]
        
        # Update dragonfly positions
        for i in range(num_dragonflies):
            # This is a simplified update rule for visualization purposes
            for j in range(num_dragonflies):
                if objective_values[j] < objective_values[i]:
                    distance = np.linalg.norm(dragonfly_positions[j] - dragonfly_positions[i])
                    attraction = np.random.uniform(-1, 1, size=num_dimensions) * distance
                    dragonfly_positions[i] += attraction
            dragonfly_positions[i] = np.clip(dragonfly_positions[i], lower_bounds, upper_bounds)

    return best_solution, history, best_optimal

def run_mfo(objective_func, bounds, num_iterations, num_moths):
    num_dimensions = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    moth_positions = np.random.uniform(lower_bounds, upper_bounds, size=(num_moths, num_dimensions))
    best_solution = None
    best_optimal = np.inf
    history = []
    
    for i in range(num_iterations):
        history.append(moth_positions.copy())
        objective_values = np.array([objective_func(pos) for pos in moth_positions])
        
        if np.min(objective_values) < best_optimal:
            best_optimal = np.min(objective_values)
            best_solution = moth_positions[np.argmin(objective_values)]
        
        flame_intensity = np.exp(-10 * i / num_iterations)
        
        # Update moth positions
        for j in range(num_moths):
            distance_to_best = np.linalg.norm(best_solution - moth_positions[j])
            step_size = np.random.uniform(-1, 1) * distance_to_best * flame_intensity
            moth_positions[j] += step_size
            moth_positions[j] = np.clip(moth_positions[j], lower_bounds, upper_bounds)
    
    return best_solution, history, best_optimal

def run_gwo(objective_func, bounds, num_iterations, num_agents):
    num_dimensions = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    positions = np.random.uniform(lower_bounds, upper_bounds, size=(num_agents, num_dimensions))
    best_solution = None
    best_optimal = np.inf
    history = []
    
    for t in range(num_iterations):
        history.append(positions.copy())
        objectives = np.array([objective_func(pos) for pos in positions])
        
        if np.min(objectives) < best_optimal:
            best_optimal = np.min(objectives)
            best_solution = positions[np.argmin(objectives)]
        
        sorted_indices = np.argsort(objectives)
        alpha_wolf = positions[sorted_indices[0]]
        beta_wolf = positions[sorted_indices[1]]
        delta_wolf = positions[sorted_indices[2]]
        
        for i in range(num_agents):
            a = 2 - 2 * t / num_iterations
            
            r1, r2 = np.random.rand(2)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha_wolf - positions[i])
            X1 = alpha_wolf - A1 * D_alpha
            
            r1, r2 = np.random.rand(2)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta_wolf - positions[i])
            X2 = beta_wolf - A2 * D_beta
            
            r1, r2 = np.random.rand(2)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta_wolf - positions[i])
            X3 = delta_wolf - A3 * D_delta
            
            positions[i] = (X1 + X2 + X3) / 3
            positions[i] = np.clip(positions[i], lower_bounds, upper_bounds)
            
    return best_solution, history, best_optimal

# --- Part 3: Streamlit App UI and Logic ---

# Title and description
st.title("Interactive Optimization Algorithm Visualizer")
st.markdown("Explore and visualize how different nature-inspired algorithms solve optimization problems.")

# Sidebar for controls
st.sidebar.header("Configuration")
algorithm_options = [
    "Genetic Algorithm", "PSO", "Simulated Annealing", 
    "TLBO", "ABC", "Cuckoo Search", "Antlion Optimization", 
    "Dragonfly Algorithm", "Moth-Flame Optimization", "Grey Wolf Optimizer"
]
algorithm_choice = st.sidebar.selectbox("Choose an Algorithm", algorithm_options)

objective_options = ["Rastrigin's Function", "Sphere Function"]
objective_choice = st.sidebar.selectbox("Choose an Objective Function", objective_options)

# Sliders for algorithm-specific parameters
num_iterations = st.sidebar.slider("Number of Iterations", 10, 500, 100)
if algorithm_choice in ["Genetic Algorithm", "PSO", "TLBO", "ABC", "Cuckoo Search", "Antlion Optimization", "Dragonfly Algorithm", "Moth-Flame Optimization", "Grey Wolf Optimizer"]:
    pop_size_label = "Population Size" if algorithm_choice not in ["Cuckoo Search", "Antlion Optimization", "Dragonfly Algorithm", "Moth-Flame Optimization", "Grey Wolf Optimizer"] else "Number of Agents"
    pop_size = st.sidebar.slider(pop_size_label, 10, 200, 50)
if algorithm_choice == "Genetic Algorithm":
    mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 1.0, 0.1)
elif algorithm_choice == "PSO":
    w = st.sidebar.slider("Inertia Weight (w)", 0.1, 1.0, 0.5)
    c1_pso = st.sidebar.slider("Cognitive Weight (c1)", 0.1, 2.0, 1.5)
    c2_pso = st.sidebar.slider("Social Weight (c2)", 0.1, 2.0, 1.5)
elif algorithm_choice == "Simulated Annealing":
    T_max = st.sidebar.slider("Max Temperature", 100.0, 10000.0, 1000.0)
    T_min = st.sidebar.slider("Min Temperature", 0.1, 10.0, 1.0)
    alpha = st.sidebar.slider("Alpha", 0.1, 1.0, 0.9)
elif algorithm_choice == "ABC":
    num_employed_bees = st.sidebar.slider("Number of Employed Bees", 5, 100, 25)
elif algorithm_choice == "Cuckoo Search":
    pa = st.sidebar.slider("Probability of Abandoning a Nest (pa)", 0.01, 1.0, 0.25)
    
# Function to run the selected algorithm
def run_optimization():
    if objective_choice == "Rastrigin's Function":
        bounds = [(-5.12, 5.12), (-5.12, 5.12)]
        func = rastrigin_function
    elif objective_choice == "Sphere Function":
        bounds = [(-100.0, 100.0), (-100.0, 100.0)]
        func = sphere_function
    else:
        st.error("Please select a valid objective function.")
        return

    st.write(f"Running **{algorithm_choice}** on **{objective_choice}**...")
    
    # Run the correct algorithm based on user selection
    if algorithm_choice == "Genetic Algorithm":
        best_solution, history, best_fitness = run_genetic_algorithm(func, bounds, num_iterations, pop_size, mutation_rate)
    elif algorithm_choice == "PSO":
        best_solution, history, best_fitness = run_pso(func, bounds, num_iterations, pop_size, w, c1_pso, c2_pso)
    elif algorithm_choice == "Simulated Annealing":
        # Note: SA history is a list of single points, not a population
        best_solution, history, best_fitness = run_sa(func, bounds, num_iterations, T_max, T_min, alpha)
    elif algorithm_choice == "TLBO":
        best_solution, history, best_fitness = run_tlbo(func, bounds, num_iterations, pop_size)
    elif algorithm_choice == "ABC":
        best_solution, history, best_fitness = run_abc(func, bounds, num_iterations, num_employed_bees)
    elif algorithm_choice == "Cuckoo Search":
        best_solution, history, best_fitness = run_csa(func, bounds, num_iterations, pop_size, pa)
    elif algorithm_choice == "Antlion Optimization":
        best_solution, history, best_fitness = run_alo(func, bounds, num_iterations, pop_size)
    elif algorithm_choice == "Dragonfly Algorithm":
        best_solution, history, best_fitness = run_dragonfly(func, bounds, num_iterations, pop_size)
    elif algorithm_choice == "Moth-Flame Optimization":
        best_solution, history, best_fitness = run_mfo(func, bounds, num_iterations, pop_size)
    elif algorithm_choice == "Grey Wolf Optimizer":
        best_solution, history, best_fitness = run_gwo(func, bounds, num_iterations, pop_size)
    else:
        st.error("Algorithm not yet implemented.")
        return
        
    st.success("Optimization Complete!")
    st.write(f"Best solution found: `x = {best_solution}`")
    st.write(f"Best fitness value: `{best_fitness:.4f}`")
    
    # Visualization
    st.subheader("Visualization of Algorithm Path")
    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func([xi, yi]) for xi in x] for yi in y])

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, opacity=0.8, colorscale="viridis", showscale=False)])
    
    # Add a trace for each generation
    for i, population_snapshot in enumerate(history):
        fig.add_trace(go.Scatter3d(
            x=population_snapshot[:, 0],
            y=population_snapshot[:, 1],
            z=[func(p) for p in population_snapshot],
            mode='markers',
            marker=dict(
                size=5,
                color=i,
                colorscale="inferno",
                cmin=0,
                cmax=len(history)
            ),
            name=f'Generation {i+1}',
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"'{algorithm_choice}' Search Path on the Objective Function Landscape",
        scene=dict(
            xaxis_title=f"X1",
            yaxis_title=f"X2",
            zaxis_title="Function Value",
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=0.5)
            )
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# The main run button
if st.sidebar.button("Run Optimization"):
    run_optimization()
