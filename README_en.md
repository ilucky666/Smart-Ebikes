

# Campus Bike-Sharing Operation & Scheduling Optimization

This project aims to build a spatiotemporal and operational model for shared bikes in a university campus, leveraging real-time parking data to optimize bike allocation, path routing, fault recovery, and operation efficiency.

## 📌 Background

While bike-sharing systems have been widely adopted on campuses, issues such as supply-demand imbalance, inefficient rebalancing, and fault accumulation still persist. This project uses GIS and Python-based analysis to provide a comprehensive solution.

## 🔧 Core Modules

- 🚲 **Bike Count Modeling**  
  Constructed a Cubic Spline + Gaussian function model to estimate the number of parked bikes at any time.

- 📈 **Demand Modeling & Dispatch Optimization**  
  Introduced the concept of supply-demand gap and used greedy algorithms to optimize bike dispatch based on real demand.

- 🧭 **Real Distance Network Modeling**  
  Built a Network Dataset in ArcGIS Pro and computed topological shortest paths using OD Cost Matrix with real path distances.

- 🔧 **Fault Detection & Route Optimization**  
  Developed a repair route planning model using an improved Ant Colony Optimization (ACO) algorithm considering fault rate and vehicle capacity.

- 🌡️ **Spatial Hotspot & Density Analysis**  
  Conducted heat map analysis, kernel density interpolation, and spatial autocorrelation based on predicted demand and parking pressure.

- 🗺️ **WebGIS Visualization** (optional)  
  Implemented a frontend interface using Gaode Map API and GNSS modules for real-time location updates, path animation, and statistics.

## 🛠️ Tools & Environment

- ArcGIS Pro 3.x
- Python 3.10 with `pandas`, `numpy`, `scipy`, `matplotlib`, `ortools`
- Jupyter Notebook
- Gaode Map API (for WebGIS)
- Markdown / LaTeX (for documentation and report)

## 📂 Project Structure

```bash
CampusBikeOptimization/
├── data/                  # Input data (Excel, shapefiles, etc.)
├── notebooks/             # Python notebooks for modeling
├── visualization/         # ArcGIS Pro files and exports
├── output/                # Model results and simulation outputs
├── webgis/                # HTML + JS files for map visualization
├── paper/                 # LaTeX files and math models
└── README.md              # Documentation
```
💡 Contact
For questions or collaboration: ethan_lu@foxmail.com

