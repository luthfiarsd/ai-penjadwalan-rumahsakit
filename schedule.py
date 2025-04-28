import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
import datetime
import copy
import pulp
import math
import random
from sklearn import tree
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="AI Penjadwalan Peserta Didik", layout="wide")


# Cache data loaders for better performance
@st.cache_data
def load_data():
    """Load data from CSV files and preprocess"""
    try:
        peserta_didik = pd.read_csv("Dummy/Data_Peserta_Didik.csv")
        patient_data = pd.read_csv("Dummy/Data_Pasien.csv")
        wahana = pd.read_csv("Dummy/Data_Wahana.csv")

        # Create simplified patient data format
        patient_counts = patient_data[["Nama Wahana", "Realita Jumlah Pasien"]].copy()

        pasien = pd.DataFrame(patient_counts)

        # Preprocessing: Extract range of normal patients
        wahana["Min_Pasien"] = (
            wahana["Jumlah Normal Pasien per Peserta Didik"]
            .str.split("-")
            .str[0]
            .str.strip()
            .astype(int)
        )
        wahana["Max_Pasien"] = (
            wahana["Jumlah Normal Pasien per Peserta Didik"]
            .str.split("-")
            .str[1]
            .str.strip()
            .astype(int)
        )

        return peserta_didik, pasien, wahana

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


# Load data
peserta_didik, pasien, wahana = load_data()
if peserta_didik is None or pasien is None or wahana is None:
    st.error("Failed to load required data. Please check your data files.")
    st.stop()

# Initialize session state if not already initialized
if "jadwal" not in st.session_state:
    st.session_state.jadwal = None
if "selected_peserta" not in st.session_state:
    st.session_state.selected_peserta = peserta_didik["ID"].tolist()
if "selected_wahana" not in st.session_state:
    st.session_state.selected_wahana = wahana["Nama Wahana"].tolist()

# Stage 1: Integer Linear Programming (ILP) for initial allocation
# def ilp_initial_allocation(peserta_list, wahana_df, patient_counts):
#     """
#     Use Integer Linear Programming to create an initial allocation that minimizes
#     deviation from normal patient-to-student ratios
#     """
#     st.info("Running Integer Linear Programming for initial allocation...")
#     progress_bar = st.progress(0)

#     # Create mapping for wahana names to indices
#     wahana_names = wahana_df['Nama Wahana'].tolist()
#     wahana_indices = range(len(wahana_names))
#     wahana_map = {name: idx for idx, name in enumerate(wahana_names)}

#     # Create ILP problem
#     prob = pulp.LpProblem("Student_Allocation", pulp.LpMinimize)

#     # Create binary variables for student allocation
#     # x[i,j] = 1 if student i is assigned to facility j
#     x = {}
#     for i in range(len(peserta_list)):
#         for j in wahana_indices:
#             x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)

#     # Create variables for deviation from normal range
#     under_dev = {}
#     over_dev = {}
#     for j in wahana_indices:
#         under_dev[j] = pulp.LpVariable(f"under_dev_{j}", lowBound=0, cat=pulp.LpContinuous)
#         over_dev[j] = pulp.LpVariable(f"over_dev_{j}", lowBound=0, cat=pulp.LpContinuous)

#     # Get minimum and maximum normal patient counts
#     min_normal = {}
#     max_normal = {}
#     for j in wahana_indices:
#         wahana_name = wahana_names[j]
#         min_normal[j] = wahana_df.loc[wahana_df['Nama Wahana'] == wahana_name, 'Min_Pasien'].values[0]
#         max_normal[j] = wahana_df.loc[wahana_df['Nama Wahana'] == wahana_name, 'Max_Pasien'].values[0]

#     # Map patient counts to wahana indices
#     patients_per_wahana = {}
#     for j in wahana_indices:
#         wahana_name = wahana_names[j]
#         patients_per_wahana[j] = patient_counts.loc[patient_counts['Nama Wahana'] == wahana_name, 'Realita Jumlah Pasien'].values[0]

#     # Add objective function: minimize total deviation
#     prob += pulp.lpSum([under_dev[j] + over_dev[j] for j in wahana_indices])

#     # Constraints:

#     # 1. Each student must be assigned to exactly one facility
#     for i in range(len(peserta_list)):
#         prob += pulp.lpSum([x[i, j] for j in wahana_indices]) == 1

#     # 2. Patient-to-student ratio constraints
#     for j in wahana_indices:
#         # Calculate total students assigned to facility j
#         students_at_j = pulp.lpSum([x[i, j] for i in range(len(peserta_list))])

#         # Calculate patients per student
#         # To avoid division by zero, we use linear constraints

#         # If under minimum range, calculate under_deviation
#         prob += min_normal[j] * students_at_j - patients_per_wahana[j] <= under_dev[j]

#         # If over maximum range, calculate over_deviation
#         prob += patients_per_wahana[j] - max_normal[j] * students_at_j <= over_dev[j]

#     # Update progress
#     progress_bar.progress(0.3)

#     # Solve the problem
#     solver = pulp.PULP_CBC_CMD(msg=False)
#     prob.solve(solver)

#     progress_bar.progress(0.7)

#     # Extract the solution
#     assignment = {}
#     allocation = {wahana_name: [] for wahana_name in wahana_names}

#     if prob.status == pulp.LpStatusOptimal:
#         for i in range(len(peserta_list)):
#             for j in wahana_indices:
#                 if pulp.value(x[i, j]) == 1:
#                     student_id = peserta_list[i]
#                     wahana_name = wahana_names[j]
#                     assignment[student_id] = wahana_name
#                     allocation[wahana_name].append(student_id)
#     else:
#         st.error("Failed to find optimal solution with ILP")

#     progress_bar.progress(1.0)
#     st.success("Integer Linear Programming allocation completed")

#     return {
#         'final': assignment,
#         'initial': allocation
#     }


# Stage 1: Simple allocation for initial distribution
def roundrobin_initial_allocation(peserta_list, wahana_df, patient_counts):

    st.info("Creating initial allocation using round-robin distribution...")
    progress_bar = st.progress(0)

    # Create mapping for wahana names
    wahana_names = wahana_df["Nama Wahana"].tolist()

    # Initialize empty allocation
    allocation = {wahana_name: [] for wahana_name in wahana_names}

    # Simple round-robin allocation
    for i, student_id in enumerate(peserta_list):
        # Allocate to wahana in circular fashion
        wahana_idx = i % len(wahana_names)
        wahana_name = wahana_names[wahana_idx]
        allocation[wahana_name].append(student_id)

        # Update progress
        progress_bar.progress((i + 1) / len(peserta_list))

    # Convert to assignment format
    assignment = {}
    for wahana_name, students in allocation.items():
        for student in students:
            assignment[student] = wahana_name

    progress_bar.progress(1.0)
    st.success("Initial allocation completed using round-robin distribution")

    return {"final": assignment, "initial": allocation}


# Stage 2: Decision Tree for status classification
def decision_tree_classification(allocation, wahana_df, patient_counts):
    """
    Use Decision Tree to classify facilities as overloaded, underutilized, or normal
    based on patient-to-student ratio
    """
    st.info("Running Decision Tree classification...")

    # Prepare data for decision tree
    wahana_names = wahana_df["Nama Wahana"].tolist()
    features = []
    labels = []

    for wahana_name in wahana_names:
        students_count = len(allocation[wahana_name]) or 1  # Avoid division by zero
        patient_count = patient_counts.loc[
            patient_counts["Nama Wahana"] == wahana_name, "Realita Jumlah Pasien"
        ].values[0]
        ratio = patient_count / students_count

        min_normal = wahana_df.loc[
            wahana_df["Nama Wahana"] == wahana_name, "Min_Pasien"
        ].values[0]
        max_normal = wahana_df.loc[
            wahana_df["Nama Wahana"] == wahana_name, "Max_Pasien"
        ].values[0]

        # Features: [ratio, min_normal, max_normal]
        features.append([ratio, min_normal, max_normal])

        # Labels: 0=underutilized, 1=normal, 2=overloaded
        if ratio < min_normal:
            labels.append(0)  # underutilized
        elif ratio > max_normal:
            labels.append(2)  # overloaded
        else:
            labels.append(1)  # normal

    # Create and train decision tree
    clf = tree.DecisionTreeClassifier(max_depth=3)

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train model
    clf.fit(features_scaled, labels)

    # Apply model to classify facilities
    status = {}
    status_counts = {"normal": 0, "overloaded": 0, "underutilized": 0}
    pasien_per_peserta = {}

    for i, wahana_name in enumerate(wahana_names):
        students_count = len(allocation[wahana_name]) or 1
        patient_count = patient_counts.loc[
            patient_counts["Nama Wahana"] == wahana_name, "Realita Jumlah Pasien"
        ].values[0]
        ratio = patient_count / students_count
        pasien_per_peserta[wahana_name] = ratio

        # Get classification
        feature = [ratio, features[i][1], features[i][2]]
        feature_scaled = scaler.transform([feature])
        classification = clf.predict(feature_scaled)[0]

        if classification == 0:
            status[wahana_name] = "underutilized"
            status_counts["underutilized"] += 1
        elif classification == 2:
            status[wahana_name] = "overloaded"
            status_counts["overloaded"] += 1
        else:
            status[wahana_name] = "normal"
            status_counts["normal"] += 1

    st.success(
        f"Decision Tree classification completed: {status_counts['normal']} normal, {status_counts['overloaded']} overloaded, {status_counts['underutilized']} underutilized"
    )

    return status, pasien_per_peserta


# # Stage 3: Simulated Annealing for optimization
# def simulated_annealing_optimization(initial_allocation, wahana_df, patient_counts, status):
#     """
#     Use Simulated Annealing to optimize student allocation when imbalances are detected
#     """
#     st.info("Running Simulated Annealing optimization...")
#     progress_bar = st.progress(0)

#     # Check if optimization is needed
#     if all(s == 'normal' for s in status.values()):
#         st.success("All facilities are in normal status. No optimization needed.")

#         # Convert initial allocation to assignment format
#         assignment = {}
#         for wahana_name, students in initial_allocation.items():
#             for student in students:
#                 assignment[student] = wahana_name

#         return {'final': assignment, 'initial': initial_allocation}

#     # Parameters for simulated annealing
#     initial_temperature = 100.0
#     final_temperature = 0.1
#     cooling_rate = 0.95
#     iterations_per_temperature = 100

#     # Extract data
#     wahana_names = wahana_df['Nama Wahana'].tolist()
#     students = []
#     for wahana_name, student_list in initial_allocation.items():
#         students.extend(student_list)

#     # Create initial solution (current state)
#     current_solution = {}
#     for wahana_name, student_list in initial_allocation.items():
#         for student in student_list:
#             current_solution[student] = wahana_name

#     # Make a copy for the best solution found
#     best_solution = current_solution.copy()

#     # Calculate initial energy (cost)
#     def calculate_energy(solution):
#         # Convert solution to allocation format
#         allocation = {wahana_name: [] for wahana_name in wahana_names}
#         for student, wahana_name in solution.items():
#             allocation[wahana_name].append(student)

#         # Calculate energy based on patient-to-student ratio
#         energy = 0
#         for wahana_name in wahana_names:
#             student_count = len(allocation[wahana_name]) or 1  # Avoid division by zero
#             patient_count = patient_counts.loc[patient_counts['Nama Wahana'] == wahana_name, 'Realita Jumlah Pasien'].values[0]
#             ratio = patient_count / student_count

#             min_normal = wahana_df.loc[wahana_df['Nama Wahana'] == wahana_name, 'Min_Pasien'].values[0]
#             max_normal = wahana_df.loc[wahana_df['Nama Wahana'] == wahana_name, 'Max_Pasien'].values[0]

#             # Penalty for deviation from normal range
#             if ratio < min_normal:
#                 energy += (min_normal - ratio) ** 2  # squared to penalize larger deviations
#             elif ratio > max_normal:
#                 energy += (ratio - max_normal) ** 2

#         return energy

#     # Get initial energy
#     current_energy = calculate_energy(current_solution)
#     best_energy = current_energy

#     # Helper function to generate a neighbor solution
#     def get_neighbor(solution):
#         neighbor = solution.copy()

#         # Randomly select a student
#         student = random.choice(students)

#         # Current wahana for this student
#         current_wahana = solution[student]

#         # Select a different wahana randomly
#         other_wahanas = [w for w in wahana_names if w != current_wahana]
#         if not other_wahanas:  # If there's only one wahana
#             return neighbor

#         new_wahana = random.choice(other_wahanas)

#         # Move student to new wahana
#         neighbor[student] = new_wahana

#         return neighbor

#     # Main simulated annealing loop
#     temperature = initial_temperature
#     total_iterations = int(math.log(final_temperature / initial_temperature, cooling_rate) * iterations_per_temperature)
#     current_iteration = 0

#     while temperature > final_temperature:
#         for _ in range(iterations_per_temperature):
#             # Generate a neighbor solution
#             neighbor_solution = get_neighbor(current_solution)

#             # Calculate neighbor energy
#             neighbor_energy = calculate_energy(neighbor_solution)

#             # Decide if we should accept the neighbor
#             if neighbor_energy < current_energy:
#                 # Accept better solution
#                 current_solution = neighbor_solution
#                 current_energy = neighbor_energy

#                 # Update best solution if needed
#                 if current_energy < best_energy:
#                     best_solution = current_solution.copy()
#                     best_energy = current_energy
#             else:
#                 # Accept worse solution with probability based on temperature
#                 delta_energy = neighbor_energy - current_energy
#                 acceptance_probability = math.exp(-delta_energy / temperature)

#                 if random.random() < acceptance_probability:
#                     current_solution = neighbor_solution
#                     current_energy = neighbor_energy

#             # Update iteration count
#             current_iteration += 1

#         # Cool down temperature
#         temperature *= cooling_rate

#         # Update progress
#         progress_percent = min(1.0, current_iteration / total_iterations)
#         progress_bar.progress(progress_percent)

#     # Ensure we've reached 100% on the progress bar
#     progress_bar.progress(1.0)

#     # Convert best solution to allocation format
#     optimized_allocation = {wahana_name: [] for wahana_name in wahana_names}
#     for student, wahana_name in best_solution.items():
#         optimized_allocation[wahana_name].append(student)

#     st.success(f"Simulated Annealing optimization completed (final energy: {best_energy:.2f})")

#     return {
#         'final': best_solution,
#         'initial': initial_allocation
#     }


def hill_climbing_optimization(initial_allocation, wahana_df, patient_counts, status):
    """
    Use Hill Climbing to optimize student allocation when imbalances are detected
    """
    st.info("Running Hill Climbing optimization...")
    progress_bar = st.progress(0)

    # Check if optimization is needed
    if all(s == "normal" for s in status.values()):
        st.success("All facilities are in normal status. No optimization needed.")

        # Convert initial allocation to assignment format
        assignment = {}
        for wahana_name, students in initial_allocation.items():
            for student in students:
                assignment[student] = wahana_name

        return {"final": assignment, "initial": initial_allocation}

    # Parameters for hill climbing
    max_iterations = 1000
    max_neighbors = 50  # Number of neighbors to evaluate per iteration

    # Extract data
    wahana_names = wahana_df["Nama Wahana"].tolist()
    students = []
    for wahana_name, student_list in initial_allocation.items():
        students.extend(student_list)

    # Create initial solution (current state)
    current_solution = {}
    for wahana_name, student_list in initial_allocation.items():
        for student in student_list:
            current_solution[student] = wahana_name

    # Make a copy for the best solution found
    best_solution = current_solution.copy()

    # Calculate energy (cost) - lower is better
    def calculate_energy(solution):
        # Convert solution to allocation format
        allocation = {wahana_name: [] for wahana_name in wahana_names}
        for student, wahana_name in solution.items():
            allocation[wahana_name].append(student)

        # Calculate energy based on patient-to-student ratio
        energy = 0
        for wahana_name in wahana_names:
            student_count = len(allocation[wahana_name]) or 1  # Avoid division by zero
            patient_count = patient_counts.loc[
                patient_counts["Nama Wahana"] == wahana_name, "Realita Jumlah Pasien"
            ].values[0]
            ratio = patient_count / student_count

            min_normal = wahana_df.loc[
                wahana_df["Nama Wahana"] == wahana_name, "Min_Pasien"
            ].values[0]
            max_normal = wahana_df.loc[
                wahana_df["Nama Wahana"] == wahana_name, "Max_Pasien"
            ].values[0]

            # Penalty for deviation from normal range
            if ratio < min_normal:
                energy += (
                    min_normal - ratio
                ) ** 2  # Squared to penalize larger deviations
            elif ratio > max_normal:
                energy += (ratio - max_normal) ** 2

        return energy

    # Get initial energy
    current_energy = calculate_energy(current_solution)
    best_energy = current_energy

    # Helper function to generate a neighbor solution
    def get_neighbor(solution):
        neighbor = solution.copy()

        # Randomly select a student
        student = random.choice(students)

        # Current wahana for this student
        current_wahana = solution[student]

        # Select a different wahana randomly
        other_wahanas = [w for w in wahana_names if w != current_wahana]
        if not other_wahanas:  # If there's only one wahana
            return neighbor

        new_wahana = random.choice(other_wahanas)

        # Move student to new wahana
        neighbor[student] = new_wahana

        return neighbor

    # Main hill climbing loop
    iteration = 0
    while iteration < max_iterations:
        # Generate and evaluate neighbors
        best_neighbor = None
        best_neighbor_energy = float("inf")

        for _ in range(max_neighbors):
            neighbor_solution = get_neighbor(current_solution)
            neighbor_energy = calculate_energy(neighbor_solution)

            # Keep track of the best neighbor
            if neighbor_energy < best_neighbor_energy:
                best_neighbor = neighbor_solution
                best_neighbor_energy = neighbor_energy

        # If we found a better neighbor, move to it
        if best_neighbor and best_neighbor_energy < current_energy:
            current_solution = best_neighbor
            current_energy = best_neighbor_energy

            # Update best solution if needed
            if current_energy < best_energy:
                best_solution = current_solution.copy()
                best_energy = current_energy
        else:
            # No better neighbor found, stop early
            break

        # Update progress
        iteration += 1
        progress_bar.progress(min(1.0, iteration / max_iterations))

    # Ensure we've reached 100% on the progress bar
    progress_bar.progress(1.0)

    # Convert best solution to allocation format
    optimized_allocation = {wahana_name: [] for wahana_name in wahana_names}
    for student, wahana_name in best_solution.items():
        optimized_allocation[wahana_name].append(student)

    st.success(
        f"Hill Climbing optimization completed (final energy: {best_energy:.2f})"
    )

    return {"final": best_solution, "initial": initial_allocation}


# Main AI scheduling function
def ai_scheduling(peserta_list, wahana_df, patient_counts):

    st.write("### Phase 1: Initial Allocation using Round-Robin (Static)")

    # Stage 1: Round-Robin for initial allocation
    roundrobin_result = roundrobin_initial_allocation(
        peserta_list, wahana_df, patient_counts
    )
    initial_allocation = roundrobin_result["initial"]

    # Stage 2: Decision Tree for classification
    st.write("### Phase 2: Status Classification using Decision Tree")
    status, ratios = decision_tree_classification(
        initial_allocation, wahana_df, patient_counts
    )

    # Stage 3: Hill Climbing for optimization
    st.write("### Phase 3: Optimization using Hill Climbing")
    final_result = hill_climbing_optimization(
        initial_allocation, wahana_df, patient_counts, status
    )

    return {"final": final_result["final"], "initial": initial_allocation}


# Helper function to visualize the initial allocation
def visualize_initial_allocation(
    initial_allocation, wahana_df, peserta_df, patient_counts=None
):
    """Visualize the initial distribution with detailed status information"""

    # Convert to DataFrame
    initial_data = []
    for wahana_name, peserta_list in initial_allocation.items():
        for peserta in peserta_list:
            initial_data.append(
                {
                    "ID": str(peserta),  # Convert to string explicitly
                    "Nama Wahana": wahana_name,
                }
            )

    initial_df = pd.DataFrame(initial_data)

    # Merge dengan data peserta untuk mendapatkan nama
    if peserta_df is not None:
        # Make sure both ID columns are strings
        peserta_df = peserta_df.copy()  # Create a copy to avoid modifying the original
        peserta_df["ID"] = peserta_df["ID"].astype(str).str.replace(",", "").str.strip()
        initial_df["ID"] = initial_df["ID"].astype(str).str.replace(",", "").str.strip()

        initial_df = pd.merge(
            initial_df,
            peserta_df[["ID", "Nama"]],
            left_on="ID",
            right_on="ID",
            how="left",
        ).drop("ID", axis=1)
        initial_df.columns = ["ID", "Nama Wahana", "Nama Peserta"]

    # Tampilkan daftar peserta dengan wahananya
    st.markdown("### Detail Penempatan Peserta")
    st.dataframe(initial_df)

    # Hitung jumlah peserta per wahana
    count_df = (
        initial_df.groupby("Nama Wahana").size().reset_index(name="Jumlah Peserta")
    )

    # Gabungkan dengan data wahana
    result_df = pd.merge(count_df, wahana_df, on="Nama Wahana")

    # Jika patient_counts tersedia, tambahkan informasi status
    if patient_counts is not None:
        # Add patient counts from the new format
        result_df = pd.merge(
            result_df,
            patient_counts[["Nama Wahana", "Realita Jumlah Pasien"]],
            on="Nama Wahana",
        )

        # Calculate patients per participant
        result_df["Pasien per Peserta"] = (
            result_df["Realita Jumlah Pasien"] / result_df["Jumlah Peserta"]
        )

        # Determine status
        result_df["Status"] = [
            (
                "normal"
                if min_val <= p <= max_val
                else ("overloaded" if p > max_val else "underutilized")
            )
            for p, min_val, max_val in zip(
                result_df["Pasien per Peserta"],
                result_df["Min_Pasien"],
                result_df["Max_Pasien"],
            )
        ]

        # Round pasien per peserta for display
        result_df["Pasien per Peserta"] = result_df["Pasien per Peserta"].round(2)

        # Calculate statistics
        total_wahana = len(result_df)
        normal_count = sum(result_df["Status"] == "normal")
        overloaded_count = sum(result_df["Status"] == "overloaded")
        underutilized_count = sum(result_df["Status"] == "underutilized")

    # Tampilkan tabel detail wahana
    st.markdown("### Detail Status Wahana")

    # Create a display DataFrame with all needed information
    display_df = result_df.copy()
    display_df["Pasien per Peserta"] = (
        display_df["Pasien per Peserta"].round(0).astype(int)
    )

    # Select columns based on whether patient_counts is available
    if patient_counts is not None:
        # Full version with status
        display_columns = [
            "Nama Wahana",
            "Jumlah Peserta",
            "Realita Jumlah Pasien",
            "Pasien per Peserta",
            "Jumlah Normal Pasien per Peserta Didik",
            "Min_Pasien",
            "Max_Pasien",
            "Status",
        ]
        # Make sure all columns exist
        display_columns = [col for col in display_columns if col in display_df.columns]
    else:
        # Simplified version without status
        display_columns = [
            "Nama Wahana",
            "Jumlah Peserta",
            "Jumlah Normal Pasien per Peserta Didik",
            "Min_Pasien",
            "Max_Pasien",
        ]
        # Make sure all columns exist
        display_columns = [col for col in display_columns if col in display_df.columns]

    # Display the table
    display_df = display_df[display_columns]

    # Apply color coding to Status column if it exists
    if "Status" in display_df.columns:

        def color_status(val):
            colors = {"normal": "green", "overloaded": "red", "underutilized": "orange"}
            color = colors.get(val, "black")
            return f"background-color: {color}; color: white"

        st.dataframe(display_df.style.map(color_status, subset=["Status"]))
    else:
        st.dataframe(display_df)

    # Display metrics
    st.markdown("### Statistik Status Wahana Awal")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Wahana Normal",
            value=f"{normal_count} ({normal_count/total_wahana*100:.1f}%)",
        )
    with col2:
        st.metric(
            label="Wahana Overloaded",
            value=f"{overloaded_count} ({overloaded_count/total_wahana*100:.1f}%)",
        )
    with col3:
        st.metric(
            label="Wahana Underutilized",
            value=f"{underutilized_count} ({underutilized_count/total_wahana*100:.1f}%)",
        )

    st.markdown("---")

    return initial_df


def generate_jadwal(selected_peserta, selected_wahana, patient_data):
    """Generate schedule using the three-stage AI approach"""
    filtered_wahana = wahana[wahana["Nama Wahana"].isin(selected_wahana)]
    filtered_patient_data = patient_data[
        patient_data["Nama Wahana"].isin(selected_wahana)
    ]

    result = ai_scheduling(selected_peserta, filtered_wahana, filtered_patient_data)
    return {"final": result["final"], "initial": result["initial"]}


def visualize_jadwal(jadwal, wahana_df, patient_data, peserta_df=None):
    """Visualize the schedule results with proper participant ID formatting and names"""
    if not jadwal:
        st.warning("No schedule to visualize")
        return

    # Convert schedule to DataFrame
    jadwal_df = pd.DataFrame(list(jadwal.items()), columns=["ID", "Nama Wahana"])

    # Format ID - remove commas and clean up
    jadwal_df["ID"] = jadwal_df["ID"].astype(str).str.replace(",", "").str.strip()

    # If peserta data is available, merge to get names
    if peserta_df is not None:
        # Clean and format ID in peserta_df
        if "ID" in peserta_df.columns:
            peserta_df["ID"] = (
                peserta_df["ID"].astype(str).str.replace(",", "").str.strip()
            )
        else:
            peserta_df["ID"] = (
                peserta_df["ID"].astype(str).str.replace(",", "").str.strip()
            )

        # Merge with participant data
        jadwal_df = pd.merge(
            jadwal_df,
            peserta_df[["ID", "Nama"]],
            left_on="ID",
            right_on="ID",
            how="left",
        )

        # Reorder and rename columns
        print(jadwal_df)
        jadwal_df = jadwal_df[["ID", "Nama", "Nama Wahana"]]
        jadwal_df.columns = ["ID", "Nama Peserta", "Nama Wahana"]
    else:
        # If no participant data, just show IDs
        jadwal_df = jadwal_df[["ID", "Nama Wahana"]]

    # Count participants per station
    peserta_per_wahana = (
        jadwal_df.groupby("Nama Wahana").size().reset_index(name="Jumlah Peserta")
    )

    # Merge with station data
    result_df = pd.merge(peserta_per_wahana, wahana_df, on="Nama Wahana")

    # Merge with patient data
    result_df = pd.merge(
        result_df,
        patient_data[["Nama Wahana", "Realita Jumlah Pasien"]],
        on="Nama Wahana",
    )

    # Calculate patients per participant
    result_df["Pasien per Peserta"] = (
        result_df["Realita Jumlah Pasien"] / result_df["Jumlah Peserta"]
    )

    # Determine status
    result_df["Status"] = [
        (
            "normal"
            if min_val <= p <= max_val
            else ("overloaded" if p > max_val else "underutilized")
        )
        for p, min_val, max_val in zip(
            result_df["Pasien per Peserta"],
            result_df["Min_Pasien"],
            result_df["Max_Pasien"],
        )
    ]

    # Round pasien per peserta for display
    result_df["Pasien per Peserta"] = (
        result_df["Pasien per Peserta"].round(0).astype(int)
    )

    # Calculate metrics
    total_wahana = len(result_df)
    normal_count = sum(result_df["Status"] == "normal")
    overloaded_count = sum(result_df["Status"] == "overloaded")
    underutilized_count = sum(result_df["Status"] == "underutilized")

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Wahana Normal",
            value=f"{normal_count} ({normal_count/total_wahana*100:.1f}%)",
        )
    with col2:
        st.metric(
            label="Wahana Overloaded",
            value=f"{overloaded_count} ({overloaded_count/total_wahana*100:.1f}%)",
        )
    with col3:
        st.metric(
            label="Wahana Underutilized",
            value=f"{underutilized_count} ({underutilized_count/total_wahana*100:.1f}%)",
        )

    # Plot distribution
    st.subheader("Distribusi Pasien per Peserta Didik")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Chart 1: Participants per station
    sns.barplot(
        x="Nama Wahana", y="Jumlah Peserta", hue="Status", data=result_df, ax=ax1
    )
    ax1.set_title("Jumlah Peserta Didik per Wahana")
    ax1.set_xticks(range(len(ax1.get_xticklabels())))
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

    # Chart 2: Patients per participant with min/max limits
    sns.barplot(
        x="Nama Wahana", y="Pasien per Peserta", hue="Status", data=result_df, ax=ax2
    )
    ax2.set_title("Jumlah Pasien per Peserta Didik")
    ax2.set_xticks(range(len(ax2.get_xticklabels())))
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

    # Draw horizontal lines for normal range
    for idx, row in result_df.iterrows():
        ax2.axhline(y=row["Min_Pasien"], color="green", linestyle="--", alpha=0.3)
        ax2.axhline(y=row["Max_Pasien"], color="red", linestyle="--", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Display detailed tables
    st.subheader("Tabel Detail Status Wahana")

    # Select columns for wahana details
    display_columns = [
        "Nama Wahana",
        "Jumlah Peserta",
        "Realita Jumlah Pasien",
        "Pasien per Peserta",
        "Jumlah Normal Pasien per Peserta Didik",
        "Min_Pasien",
        "Max_Pasien",
        "Status",
    ]

    # Make sure all columns exist
    display_columns = [col for col in display_columns if col in result_df.columns]
    display_df = result_df[display_columns]

    # Apply color coding to Status column
    def color_status(val):
        colors = {"normal": "green", "overloaded": "red", "underutilized": "orange"}
        color = colors.get(val, "black")
        return f"background-color: {color}; color: white"

    st.dataframe(display_df.style.map(color_status, subset=["Status"]))

    # Display detail jadwal peserta
    st.subheader("Detail Jadwal Peserta Didik")
    st.dataframe(jadwal_df)


def visualize_optimization_results(initial_status, final_status):
    """
    Create a side-by-side comparison of station status before and after optimization
    """
    # Convert status dictionaries to counts
    initial_counts = {"normal": 0, "overloaded": 0, "underutilized": 0}
    final_counts = {"normal": 0, "overloaded": 0, "underutilized": 0}

    for wahana, status in initial_status.items():
        initial_counts[status] += 1

    for wahana, status in final_status.items():
        final_counts[status] += 1

    # Create DataFrame for visualization
    comparison_data = pd.DataFrame(
        {
            "Status": ["Normal", "Overloaded", "Underutilized"],
            "Initial": [
                initial_counts["normal"],
                initial_counts["overloaded"],
                initial_counts["underutilized"],
            ],
            "Final": [
                final_counts["normal"],
                final_counts["overloaded"],
                final_counts["underutilized"],
            ],
        }
    )

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set bar positions
    bar_width = 0.35
    r1 = np.arange(len(comparison_data))
    r2 = [x + bar_width for x in r1]

    # Create bars
    bars1 = ax.bar(
        r1,
        comparison_data["Initial"],
        width=bar_width,
        label="Initial",
        color="skyblue",
    )
    bars2 = ax.bar(
        r2,
        comparison_data["Final"],
        width=bar_width,
        label="Final (After Optimization)",
        color="salmon",
    )

    # Add labels and title
    ax.set_xlabel("Status")
    ax.set_ylabel("Number of Stations (Wahana)")
    ax.set_title("Comparison of Station Status Before and After Optimization")
    ax.set_xticks([r + bar_width / 2 for r in range(len(comparison_data))])
    ax.set_xticklabels(comparison_data["Status"])
    ax.legend()

    # Add text labels on bars
    for i, bars in enumerate([bars1, bars2]):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                str(int(height)),
                ha="center",
                va="bottom",
            )

    # Display the chart
    st.pyplot(fig)


def display_optimization_tables(
    initial_assignment, final_assignment, wahana_df, patient_data, peserta_df
):
    """
    Display tables showing the before and after status of the optimization
    """

    # Function to calculate station status
    def calculate_status(assignment):
        # Convert assignment to allocation format
        allocation = {wahana_name: [] for wahana_name in wahana_df["Nama Wahana"]}
        for student_id, wahana_name in assignment.items():
            if wahana_name in allocation:
                allocation[wahana_name].append(student_id)

        # Calculate status for each station
        status = {}
        ratios = {}

        for wahana_name, students in allocation.items():
            if wahana_name not in wahana_df["Nama Wahana"].values:
                continue

            students_count = len(students) or 1  # Avoid division by zero

            # Get patient count for this station
            patient_row = patient_data[patient_data["Nama Wahana"] == wahana_name]
            if patient_row.empty:
                continue

            patient_count = patient_row["Realita Jumlah Pasien"].values[0]
            ratio = patient_count / students_count
            ratios[wahana_name] = ratio

            # Get normal range for this station
            wahana_row = wahana_df[wahana_df["Nama Wahana"] == wahana_name]
            if wahana_row.empty:
                continue

            min_normal = wahana_row["Min_Pasien"].values[0]
            max_normal = wahana_row["Max_Pasien"].values[0]

            # Determine status
            if ratio < min_normal:
                status[wahana_name] = "underutilized"
            elif ratio > max_normal:
                status[wahana_name] = "overloaded"
            else:
                status[wahana_name] = "normal"

        return status, ratios

    # Calculate status for initial and final assignments
    initial_status, initial_ratios = calculate_status(initial_assignment)
    final_status, final_ratios = calculate_status(final_assignment)

    # Create comparison table of station status
    status_comparison = []
    for wahana_name in wahana_df["Nama Wahana"]:
        if wahana_name in initial_status and wahana_name in final_status:
            status_comparison.append(
                {
                    "Nama Wahana": wahana_name,
                    "Rasio Awal": round(initial_ratios.get(wahana_name, 0), 2),
                    "Status Awal": initial_status.get(wahana_name, "N/A"),
                    "Rasio Akhir": round(final_ratios.get(wahana_name, 0), 2),
                    "Status Akhir": final_status.get(wahana_name, "N/A"),
                }
            )

    status_df = pd.DataFrame(status_comparison)

    # Create participant assignment comparison
    # Get participant names
    id_to_name = {}
    if peserta_df is not None:
        for _, row in peserta_df.iterrows():
            id_to_name[str(row["ID"]).strip()] = row["Nama"]

    # Find changed assignments
    changes = []
    for student_id in set(initial_assignment.keys()) & set(final_assignment.keys()):
        initial_wahana = initial_assignment.get(student_id)
        final_wahana = final_assignment.get(student_id)

        if initial_wahana != final_wahana:
            changes.append(
                {
                    "ID": student_id,
                    "Nama Peserta": id_to_name.get(str(student_id).strip(), "Unknown"),
                    "Wahana Awal": initial_wahana,
                    "Wahana Akhir": final_wahana,
                }
            )

    changes_df = pd.DataFrame(changes)

    # Display the tables
    st.subheader("Perbandingan Status Wahana (Sebelum vs Sesudah Optimasi)")

    # Apply styling to status columns
    def color_status(val):
        colors = {"normal": "green", "overloaded": "red", "underutilized": "orange"}
        color = colors.get(val, "black")
        return f"background-color: {color}; color: white"

    if not status_df.empty:
        st.dataframe(
            status_df.style.map(color_status, subset=["Status Awal", "Status Akhir"])
        )
    else:
        st.warning("No station status comparison data available.")

    # Display participant changes
    st.subheader("Perubahan Penempatan Peserta Didik")
    if not changes_df.empty:
        st.dataframe(changes_df)
        st.info(
            f"{len(changes_df)} dari {len(initial_assignment)} peserta didik dipindahkan dalam proses optimasi."
        )
    else:
        st.success("Tidak ada perubahan penempatan peserta didik yang diperlukan.")

    # Return status dictionaries for chart creation
    return initial_status, final_status


# Main app layout
def main():
    st.title("AI Penjadwalan Peserta Didik")

    # Display tabs for app sections
    tab1, tab2, tab3 = st.tabs(["Data & Filter", "Jadwal", "Tentang Aplikasi"])

    with tab1:
        st.header("Data dan Filter")

        # 1. Data Visualization Section
        st.subheader("Tampilan Data")
        st.markdown("#### Data Wahana")
        st.dataframe(
            wahana[
                [
                    "Nama Wahana",
                    "Kapasitas",
                    "Estimasi Pasien Harian",
                    "Jumlah Normal Pasien per Peserta Didik",
                ]
            ]
        )

        # Process the patient data from the CSV

        patient_data = pd.read_csv("Dummy/Data_Pasien.csv")

        patient_counts = patient_data[["Nama Wahana", "Realita Jumlah Pasien"]].copy()

        pasien_df = pd.DataFrame(patient_counts)

        st.markdown(f"#### Data Pasien")
        st.dataframe(pasien_df)

        st.markdown("#### Data Peserta Didik")
        st.dataframe(peserta_didik[["ID", "Nama"]])

        # 2. Filter Section
        st.subheader("Filter Data")

        # Filter for participants
        st.markdown("#### Pilih Peserta Didik")
        all_peserta = st.checkbox("Pilih Semua Peserta Didik", value=True)

        if all_peserta:
            # Select all peserta
            selected_peserta = peserta_didik["ID"].tolist()
        else:
            # Allow individual selection
            selected_peserta = st.multiselect(
                "Pilih ID:",
                options=peserta_didik["ID"].tolist(),
                default=peserta_didik["ID"].iloc[:5].tolist(),  # Default to first 5
            )

        # Filter for wahana
        st.markdown("#### Pilih Wahana")
        all_wahana = st.checkbox("Pilih Semua Wahana", value=True)

        if all_wahana:
            # Select all wahana
            selected_wahana = wahana["Nama Wahana"].tolist()
        else:
            # Allow individual selection
            selected_wahana = st.multiselect(
                "Pilih Nama Wahana:",
                options=wahana["Nama Wahana"].tolist(),
                default=wahana["Nama Wahana"].iloc[:5].tolist(),  # Default to first 5
            )

        # Save selected filters to session state
        st.session_state.selected_peserta = selected_peserta
        st.session_state.selected_wahana = selected_wahana

        # Generate button
        if st.button("Generate Jadwal", type="primary"):
            with st.spinner("Memproses jadwal..."):
                st.session_state.jadwal = generate_jadwal(
                    selected_peserta, selected_wahana, pasien_df
                )

            st.success(
                f"Jadwal berhasil dibuat untuk {len(selected_peserta)} peserta didik di {len(selected_wahana)} wahana!"
            )

            # Switch to the Jadwal tab
            st.query_params["active_tab"] = "Jadwal"

    with tab2:
        st.header("Hasil Penjadwalan")

        if st.session_state.jadwal:
            # Visualize initial allocation
            st.subheader("Alokasi Awal (Integer Linear Programming)")
            initial_allocation = st.session_state.jadwal["initial"]

            # Convert allocation format to assignment format for visualization
            initial_assignment = {}
            for wahana_name, students in initial_allocation.items():
                for student in students:
                    initial_assignment[student] = wahana_name

            # Show initial assignment
            initial_df = visualize_jadwal(
                initial_assignment, wahana, pasien_df, peserta_didik
            )

            # Final allocation after optimization
            st.markdown("---")
            st.subheader("Alokasi Final (Setelah Optimasi)")
            final_assignment = st.session_state.jadwal["final"]

            # Show final assignment
            visualize_jadwal(final_assignment, wahana, pasien_df, peserta_didik)

            # Add side-by-side comparison
            st.markdown("---")
            st.subheader("Analisis Perbandingan Sebelum dan Sesudah Optimasi")

            # Display detailed comparison tables
            initial_status, final_status = display_optimization_tables(
                initial_assignment, final_assignment, wahana, pasien_df, peserta_didik
            )

            # Show comparative chart
            visualize_optimization_results(initial_status, final_status)
        else:
            st.info("Silahkan generate jadwal terlebih dahulu di tab 'Data & Filter'.")

    with tab3:
        st.header("Tentang Aplikasi")
        st.write(
            """
        ### AI Penjadwalan Peserta Didik
        
        Aplikasi ini menggunakan tiga algoritma untuk menghasilkan jadwal optimal:
        
        1. **Round-Robin**: Membuat alokasi awal secara statis (kondisi belum optimal)
        2. **Decision Tree**: Mengklasifikasikan wahana sebagai normal, overloaded, atau underutilized.
        3. **Hill Climbing**: Jika ada ketidakseimbangan, optimasi dilakukan untuk merelokasi peserta didik.
        
        #### Dataset:
        - **Data Wahana**: Informasi tentang wahana, kapasitas, dan batas jumlah pasien normal.
        - **Data Pasien**: Jumlah pasien aktual di setiap wahana.
        - **Data Peserta Didik**: Informasi tentang peserta didik yang akan dijadwalkan.
        
        #### Output:
        - Jadwal awal hasil inisialisasi
        - Status wahana hasil Decision Tree
        - Jadwal final hasil optimasi
        """
        )

        st.markdown("---")
        st.markdown("© 2025 - P AI")


# Run the app
if __name__ == "__main__":
    main()
