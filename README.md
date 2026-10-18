# Computational Solid Mechanics Repository

This repository contains a collection of Python scripts, models, and computational tools for structural mechanics, finite element analysis (FEA), and advanced solid mechanics simulations. The calculations and models are primarily implemented using **Python**, leveraging the **FEniCSx** computing platform for solving partial differential equations (PDEs) and **Gmsh** for complex mesh generation.

## 🛠️ Tools & Technologies
* **Python**: Primary programming language used across all modules.
* **FEniCSx**: Advanced finite element computing platform used for structural mechanics and physics simulations.
* **Gmsh**: 3D finite element mesh generator used for complex geometries.

## 📂 Repository Structure

The repository is organized into the following directories based on their specific physical models and methodologies:

* **`Cohesive zone model/`**
  Contains models for describing materials in the presence of interfaces. It includes a mixed-mode opening model that takes into account both tension and shear behaviors.

* **`FEM structural mechanics classic/`**
  Dedicated to classical finite element method (FEM) computations. It includes linear elasticity calculations for beams and columns featuring complex geometries.

* **`Gmsh geometries/`**
  A collection of meshes generated with Gmsh. These mesh files are prepared and formatted specifically for FEM calculations within FEniCSx.

* **`Physic informed neural network/`**
  Contains Physics-Informed Neural Network (PINN) models applied to the study of structures in static conditions.

* **`Plasticity/`**
  Focuses on non-linear material behavior, specifically plasticity models. Includes implementations for **Drucker-Prager** and **Von Mises** yield criteria.

* **`Topology optimisation/`**
  Contains scripts and models for the structural topology optimization of mechanical components.

* **`Shear Analytical/`**
  A dedicated folder for the analytical calculations of shear stresses for a beam in bending.

## 🚀 Getting Started

To run the scripts in this repository, ensure you have a working Python environment with FEniCSx and Gmsh installed. 

*(You can add specific installation instructions for your environment, Docker commands, or Conda environment setups here if necessary).*
