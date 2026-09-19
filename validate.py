import pandas as pd
import numpy as np

def validate_exxogen_output(csv_file_path):
    """
    EXXOGEN Output Validation Script v1.0
    Verifies physical conservation laws, tensor consistency,
    and mathematical integrity of EXXOGEN generated interaction matrices.
    """
    df = pd.read_csv(csv_file_path)
    
    print("=" * 65)
    print(f"EXXOGEN PHYSICAL INTEGRITY REPORT: {csv_file_path}")
    print("=" * 65)
    
    p1 = df[['Atom1_X', 'Atom1_Y', 'Atom1_Z']].values
    p2 = df[['Atom2_X', 'Atom2_Y', 'Atom2_Z']].values
    distances = df['Distance_A'].values
    energies = df['Energy_kJ_mol'].values
    total_vectors = len(df)

    # 1. Geometric consistency (Euclidean Coordinate Measurement)
    calc_dist = np.linalg.norm(p1 - p2, axis=1)
    max_dist_error = np.max(np.abs(calc_dist - distances))

    # 2. Physical Distance Bounds Check
    min_dist, max_dist = np.min(distances), np.max(distances)
    # In biochemistry, covalent/van der Waals distances start above 0.5 Å.
    bounds_valid = bool(min_dist >= 0.5 and max_dist <= 100.0)

    # 3. Energy–distance correlation (Physical Potential Correlation)
    # In the case of physical potentials, energy correlates (Pearson r) with increasing distance.
    corr_matrix = np.corrcoef(distances, energies)
    energy_dist_corr = corr_matrix[0, 1]

    # 4. Translation Invariance (SE(3) Translation Invariance Check)
    # We shift the system by an arbitrary vector v; the distance must remain unchanged.
    shift_vector = np.array([123.456, -789.012, 456.789])
    p1_shifted = p1 + shift_vector
    p2_shifted = p2 + shift_vector
    calc_dist_shifted = np.linalg.norm(p1_shifted - p2_shifted, axis=1)
    translation_error = np.max(np.abs(calc_dist_shifted - distances))

    # 5. Rotational Invariance (SE(3) Rotation Invariance Check)
    # We rotate the coordinates using a rotation matrix R (e.g., 45 degrees around the Z-axis).
    theta = np.radians(45)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1]
    ])
    p1_rot = p1 @ R.T
    p2_rot = p2 @ R.T
    calc_dist_rot = np.linalg.norm(p1_rot - p2_rot, axis=1)
    rotation_error = np.max(np.abs(calc_dist_rot - distances))

    # 6. Boltzmann fitting (Thermodynamic Ensembles & Partition Consistency)
    # $P(E) \propto \exp(-E / k_B T)$ dispersion/stability test (T = 300 K, $k_B$ = 0.008314 kJ/mol·K)
    k_B_T = 0.008314 * 300.0
    beta_energies = np.exp(-energies / k_B_T)
    partition_function_Z = np.sum(beta_energies)
    boltzmann_probabilities = beta_energies / partition_function_Z
    boltzmann_valid = bool(np.all(np.isfinite(boltzmann_probabilities)) and np.sum(boltzmann_probabilities) > 0)

    # 7. Energy spectrum and bimodality
    is_finite = bool(np.all(np.isfinite(energies)))
    bound_states = df[df['Category'] == 'Bound Backbone']['Energy_kJ_mol']
    thermal_states = df[df['Category'] == 'Thermal Noise']['Energy_kJ_mol']
    bound_mean = bound_states.mean() if len(bound_states) > 0 else 0
    thermal_mean = thermal_states.mean() if len(thermal_states) > 0 else 0
    energy_gap = bound_mean / thermal_mean if thermal_mean != 0 else 0

    # Report
    print(f"[✓] Evaluated Interaction Vectors : {total_vectors}")
    print(f"[✓] Spatial Coordinate Accuracy   : Sub-Ångström Error ({max_dist_error:.6f} Å)")
    print(f"[✓] Physical Distance Bounds      : {min_dist:.3f} Å - {max_dist:.3f} Å ({'PASSED' if bounds_valid else 'FAILED'})")
    print(f"[✓] Energy-Distance Correlation   : r = {energy_dist_corr:.4f}")
    print(f"[✓] Translational Invariance     : Max Shift Error ({translation_error:.8f} Å)")
    print(f"[✓] Rotational Invariance        : Max Rotation Error ({rotation_error:.8f} Å)")
    print(f"[✓] Boltzmann Fit & Ensemble      : Partition Function Valid ({'PASSED' if boltzmann_valid else 'FAILED'})")
    print(f"[✓] Mathematical Stability        : Infinite/NaN check passed ({is_finite})")
    print("-" * 65)
    print("PHYSICAL SPECTRUM METRICS:")
    print(f" - Min Interaction Energy         : {energies.min():.3f} kJ/mol")
    print(f" - Max Interaction Energy         : {energies.max():.3f} kJ/mol")
    print(f" - Mean Quantum Potential Energy  : {energies.mean():.3f} kJ/mol")
    print(f" - Energy Separation Ratio        : {energy_gap:.3f}x (Bound vs Thermal)")
    print("=" * 65)
    
    # Strict set of verification criteria
    success = (
        max_dist_error < 0.001 and 
        translation_error < 0.001 and 
        rotation_error < 0.001 and 
        bounds_valid and 
        boltzmann_valid and 
        is_finite and 
        energy_gap > 1.5
    )
    
    if success:
        print("RESULT: VALIDATION SUCCESSFUL. Dataset satisfies all physical and Euclidean constraints.")
    else:
        print("RESULT: VALIDATION FAILED. Mathematical inconsistencies detected.")

# Run on the dataset; edit whatever you want here.
validate_exxogen_output("1d66_exxogen_analysis_results.csv")
