# CS633-Project

## Overview
This project implements a high-performance, parallel data processing engine designed to analyze large-scale 3D time-series datasets. Built using **C** and **MPI (Message Passing Interface)**, it efficiently computes local minima, local maxima, and global extrema across massive 3D volumes.

The engine features two distinct I/O implementations to demonstrate performance trade-offs:
1.  **Sequential I/O:** Master process reads and scatters data.
2.  **Parallel I/O:** All processes read their respective data chunks simultaneously using `MPI_File` views.

## Features
* **Scalable Parallelization:** Uses 3D domain decomposition to distribute workload across a $P_x \times P_y \times P_z$ process grid.
* **Non-Blocking Communication:** Utilizes `MPI_Isend` and `MPI_Irecv` for efficient boundary data exchange between neighboring processes.
* **Optimized I/O:** Includes a Parallel I/O implementation (`main_io.c`) utilizing `MPI_File_set_view` and `MPI_Type_create_hindexed` for significant speedups on large datasets.
* **Global Reduction:** Efficiently aggregates local results to compute global statistics using `MPI_Reduce`.

## Project Structure
* `src_seq.c`: Source code using Sequential I/O (Master read + Scatter).
* `src_par.c`: Source code using Parallel I/O (MPI-IO).
* `report.pdf`: Detailed performance analysis, scaling plots, and implementation logic.
* `assignment.pdf`: Original problem statement and constraints.

## Compilation
This project requires an MPI implementation (e.g., MPICH, OpenMPI).

**For Sequential I/O:**
```bash
mpicc -o engine_seq src_seq.c
