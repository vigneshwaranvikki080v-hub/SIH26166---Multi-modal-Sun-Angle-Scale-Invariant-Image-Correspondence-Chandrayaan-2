# Chandrayaan-2 × NASA LRO NAC Lunar Image Registration
### SIH Problem Statement SIH26166 | Theme: Space Technology | Organization: ISRO

An autonomous dual-matcher cross-sensor registration platform designed to register Chandrayaan-2 optical imagery (OHRC / TMC-2) with NASA Lunar Reconnaissance Orbiter Narrow Angle Camera (LRO NAC) reference images across extreme illumination, scale, and sensor variations.

---

## 🌐 Live Demo

**Try the Interactive Lunar Registration Demo:**  
🔗 [**View Live Demo**](https://github.com/vigneshwaranvikki080v-hub/SIH26166---Multi-modal-Sun-Angle-Scale-Invariant-Image-Correspondence-Chandrayaan-2/tree/main)

Explore real-time cross-sensor image registration with interactive visualization tools:
- **Tie-points matching** with color-coded inliers/outliers
- **Split wipe slider** for seamless boundary alignment inspection
- **Checkerboard overlay** for crater rim continuity verification
- **Difference heatmaps** showing sub-pixel accuracy metrics
- **Export capabilities** for registration results and metrics

---

## 📁 Project Structure

```
CHANDRAYAAN - 2 SIH/
├── index.html                   # Interactive Web Application & Visualizer
├── style.css                    # Modern Obsidian/Lunar Dark Styling & Glassmorphism
├── app.js                       # Interactive Canvas, Wipe Slider, Inliers Table & CSV Exporter
├── server.py                    # Lightweight Python Backend with OpenCV /api/register Endpoint
├── run_website.bat              # One-Click Windows Launcher
├── prepare_data.py              # Script to extract authentic crops from Chandrayaan-2 swath
├── README.md                    # Project Documentation & Setup Guide
└── assets/
    ├── loc1_ohrc.png            # Location 1 Chandrayaan-2 OHRC image
    ├── loc1_lronac.png          # Location 1 NASA LRO NAC reference image
    ├── loc1_registered.png      # Location 1 registered composite output
    ├── loc2_ohrc.png            # Location 2 OHRC image (steep shadow terrain)
    ├── loc2_lronac.png          # Location 2 LRO NAC reference image
    ├── loc2_registered.png      # Location 2 registered composite output
    ├── ch2_swath_preview.png    # Authentic Chandrayaan-2 swath preview
    ├── match_points_loc1.csv    # Location 1 verified inlier tie-points dataset
    ├── match_points_loc2.csv    # Location 2 verified inlier tie-points dataset
    ├── metrics_loc1.json        # Location 1 quantitative metrics report
    └── metrics_loc2.json        # Location 2 quantitative metrics report
```

---

## 🚀 How to Run the Project

### Option 1: Standalone / Offline Mode (No Setup Required)
Simply double-click `index.html` in Windows File Explorer. It opens instantly in Google Chrome, Microsoft Edge, or Firefox.

### Option 2: Live Python Backend Mode (Recommended)
Double-click `run_website.bat` (or run `python server.py` in your terminal).
- Starts a local HTTP server on `http://localhost:5000`
- Automatically opens your web browser
- Enables live OpenCV registration on custom uploaded images via `/api/register`

---

## 🌟 Key Capabilities

1. **Dual-Matcher Architecture**:
   - **RIFT (Radiation-Variation Insensitive Feature Transform)**: Uses frequency-domain phase congruency to extract structural features (craters, ridges, rocks) completely independent of lighting and shadow variations.
   - **LoFTR (Local Feature Transformer)**: Uses deep multiscale self and cross-attention transformers to bridge resolution and viewpoint disparities.
2. **Bucket-Capping (Uniform Grid Distribution)**:
   - Partitions images into a 4×4 spatial grid, capping points per cell to prevent corner clustering and satisfy ISRO's uniform correspondence requirement.
3. **Four Interactive Visual Modes**:
   - **Tie-Points Match Canvas**: Side-by-side view with interactive correspondence lines (Green for inliers, Red for outliers) and coordinate inspection tooltips.
   - **Split Wipe Slider**: Draggable handle to visually inspect boundary alignment between the registered image and the reference image.
   - **Checkerboard Alignment**: Interleaved alternating squares to verify crater rim continuity across tile borders.
   - **Difference Residual Heatmap**: Pixel-level error map showing sub-pixel accuracy.
4. **Verified Inliers Table & CSV Integration**:
   - Live interactive table with status filters and search.
   - One-click **Download Verified Inliers CSV** button.
   - One-click **Download Registered Image (PNG)** and **Export Metrics (JSON)**.
5. **Empirical South Pole Benchmarks**:
   - **Location 1**: 6 inliers, 3.12 px RMSE (LoFTR & RIFT consensus cross-validation).
   - **Location 2**: RIFT 10 inliers vs. LoFTR 6 inliers (+66.7% gain under severe shadows), 4.39 px RMSE.

---

## 📦 Requirements (Optional for Live Backend)
- Python 3.8+
- `opencv-python`
- `numpy`
*(If running purely via `index.html`, no Python or libraries are needed at all!)*
