# ADR-0001: Three-layer architecture

Status: Accepted

RadCounterSim is independent rather than an OceanSim fork. The pure Python
`radcounter.core` owns domain logic. `radcounter.radiation.native` owns Embree
transport. `radcounter.isaac` owns USD, UI, physics, robots, and ROS adapters.
Dependency arrows point toward core contracts only.
