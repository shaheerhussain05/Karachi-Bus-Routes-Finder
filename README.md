# 🚍 Karachi Bus Route Finder

A desktop-based application built with **Python**, **Tkinter (ttkbootstrap)**, **Folium**, and **PIL** that finds the shortest and most efficient bus route between two locations in Karachi. This project supports intelligent routing via **A\*** search with recovery, **BFS**, and an optional **Evolutionary Algorithm**.

---

## 🧭 Features

- Interactive GUI to select source and destination
- Displays the route and transitions (bus names, recovery jumps)
- Opens an HTML-based map (via Folium) showing your route visually
- Uses real Karachi public transport data (bus stops, connections)
- Intelligent routing algorithms:
  - A\* Search (with optional bus-switch penalty and fallback recovery)
  - Breadth-First Search (BFS)
  - Evolutionary Search (optional and experimental)
