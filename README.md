# Interactive Nature-Inspired Optimization Algorithm Visualizer | BTech Project

This project is an interactive web application that provides a platform for comparing and visualizing the performance of various nature-inspired optimization algorithms. Developed using Streamlit, it serves as both a tool for understanding complex optimization concepts and a portfolio piece demonstrating proficiency in algorithm implementation, data visualization, and web application development.

## 🌟 Features

* **10 Implemented Algorithms:** The application includes a modular framework with implementations of a diverse set of optimization algorithms:

  * Genetic Algorithm (GA)

  * Particle Swarm Optimization (PSO)

  * Simulated Annealing (SA)

  * Teaching-Learning-Based Optimization (TLBO)

  * Artificial Bee Colony (ABC)

  * Cuckoo Search Algorithm (CSA)

  * Antlion Optimization (ALO)

  * Dragonfly Algorithm (DA)

  * Moth-Flame Optimization (MFO)

  * Grey Wolf Optimizer (GWO)

* **Interactive Visualization:** Users can select an algorithm and objective function, and the application generates a real-time 3D plot showing the algorithm's search path on the function's landscape. This dynamic visualization helps to intuitively understand the exploration and exploitation trade-offs of each method.

* **Customizable Parameters:** The sidebar allows for easy adjustment of key algorithm parameters (e.g., population size, number of iterations, mutation rate), enabling users to experiment and observe how these changes impact performance.

* **Clear Results:** The application displays the best solution found and its corresponding fitness value, providing a quantitative measure of each algorithm's effectiveness.

## 🚀 How to Run Locally

To run this application on your local machine, follow these simple steps:

1.  **Clone the Repository**

    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install Dependencies**
    You need to have Python and `pip` installed. The required packages are listed in a `requirements.txt` file (you can create one from the code by noting the imports).

    ```bash
    pip install streamlit numpy plotly
    ```

3.  **Run the App**

    ```bash
    streamlit run app.py
    ```

The application will automatically open in your default web browser at `http://localhost:8501`.

## 📚 Core Concepts & Technologies

* **Python:** The core language for all algorithms and the web application.

* **NumPy:** Used for efficient numerical computations and handling of algorithm populations.

* **Streamlit:** A powerful, easy-to-use framework for creating beautiful, interactive web applications for data science and machine learning projects with pure Python.

* **Plotly:** A versatile graphing library for creating the interactive 3D visualizations.

