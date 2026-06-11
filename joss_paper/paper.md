---
title: 'VIMES: VIsualization of Massive Evolving Stars'
tags:
  - Python
  - astronomy
  - gravitational waves
  - binary evolution
  - visualization
authors:
  - name: Laya Binu
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Floor Broekgaarden 
    affiliation: 1
  - name: Amedeo Romagnolo
    affiliation: 2

affiliations:
 - name: Department of Astronomy and Astrophysics, University of California San Diego, 9500 Gilman Drive, La Jolla, CA 92093, USA
   index: 1
 - name: 
   index: 2

date: xx Month 2026
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/1538-4365/ac416c
aas-journal: Astrophysical Journal Supplements 
---

# Summary

Binary stellar evolution describes the complex life of two gravitationally bound stars, whose initial conditions and binary interactions (including mass transfer and common-envelope episodes) ultimately determine the system's outcome and whether it produces specific systems such as compact object mergers detectable as gravitational waves. 
Rapid binary population synthesis codes enable large-scale simulations of these systems, calculating detailed time-series data describing the evolving physical properties of each star and the binary properties. While indispensable for theoretical studies, these numerical outputs can be difficult to interpret intuitively or communicate to a general scientific audience.

VIMES (VIsualization of Massive Evolving Stars) is a Python package that converts COMPAS detailed output files intoinsightful animations of binary stellar evolution. Each animation represents the accurate information for the evolving star;s radius, binary separationm color (effective temperature), eccentricity, and evolutionary phase at every timestep.   VIMES handles the full range of binary configurations produced by a population synthesis code such as COMPAS, and generates a 3D animation. By bridging the gap between raw simulation data and an intuitive visual representation, VIMES makes binary stellar evolution accessible for presentations, publications, and outreach.


# Statement of need

Rapid population synthesis codes such as COMPAS [@Riley:2022, @TeamCompas:2025], COSMIC [@Breivik:2020], and SEVN [@Iorio:2023] are the primary tools for studying the evolution of binary stars and their formation pathways to exotic transients such as compact binary systems and their gravitational wave signatures. These codes produce data files containing numerical values for hundreds of physical quantities across thousands of timesteps. Although standard static diagrams:  HR diagrams, Kippenhahn diagrams, and binary cartoon figures are widely used to summarize the outcomes of such simulations, they are sometimes challenging to infer, are often hand-drawn schematic cartoons, and do often not capture the full continuous evolution of a binary system but instead illustrate only the key evolutionary phases of a binary in a static and qualitative way.
 Creating an accurate figure for a specific simulated binary is time-consuming, and often reqyures omitting the evolutionary transitions between phases that are often scientifically important. Recently, Tom Wagg created with cogsworth a tool to create such static cartoons in 2D, but an open-source tool that visualizes this in 3D and as a continuous evolution and that automates the creation of accurate, system-specific binary evolution from population synthesis code is still lacking .
VIMES addresses this gap. It is designed for researchers who study binary stellar evolution and need a quick, accurate way to visualize the evolution of a specific simulated system — whether to build intuition during analysis or to communicate results in
presentations and outreach. The animations produced are quantitatively accurate to the underlying simulation: all spatial scales, color temperatures, and evolutionary phases correspond directly to the data, rather than being schematic approximations.
VIMES allows for any binary system evolved with COMPAS to be turned into an animation, with the user having a choice over the type of images used for the visualization, as well as the type of scaling used when converting the data into an animation. 


# Implementation 
VIMES processes detailed output files from population synthesis codes such as COMPAS and StarTrack in two stages: data ingestion and animation rendering.

## Data Ingestion and Frame Construction
VIMES reads the binary's evolution from a COMPAS HDF5 detailed output file, extracting time-series data for both stars and the orbit. The data are first segmented into
evolutionary phases, defined by changes in the stellar type of either star (using the stellar type classification scheme of @Hurley:2000) or the onset of mass transfer. This
phase structure is used to determine frame sampling: rather than allocating frames proportional to time spent in each phase (which would result in animations dominated
by long, visually uneventful main-sequence phases), VIMES samples a fixed number of frames from each phase. This ensures that short but physically important phases — such
as common-envelope episodes or supernova kicks — receive adequate representation in the animation.
Where adjacent timesteps show large fractional or absolute changes in any displayed quantity, VIMES inserts linearly interpolated intermediate frames to ensure smooth visual transitions. The processed frames are cached as a compressed .npz file, decoupling data preparation from rendering and allowing the animation to be regenerated with different visual settings without re-reading the population synthesis output.


## Animation Rendering
The second stage reads the cached frame data and renders each frame as a two-dimensional representation of the binary system, with the two stars drawn at their correct relative sizes (proportional to stellar radius) and the correct orbital separation and eccentricity.
The orbit is drawn as an ellipse with the correct eccentricity, and the positions of the stars along the orbit are updated frame by frame. VIMES supports two visual modes:

 - Default mode
renders stars using a set of cartoon-style images that change
discretely with stellar type, making phase transitions immediately legible to an
audience unfamiliar with the underlying data conventions.
 - TULIPS mode
maps the effective surface temperature of each star to a
physically motivated RGB color using the temperature-to-color conversion from
the TULIPS package [@Laplace:2022], producing a more quantitatively accurate
visual at the cost of reduced contrast between evolutionary phases.

The rendered frames are assembled into a video file using matplotlib's animation framework [@Hunter:2007]. Figure 1 shows example snapshots from an animation of a double compact object progenitor system at four key evolutionary phases.



![Example snapshots from a VIMES animation of a massive binary system. From left to
right: (a) the initial main-sequence phase, (b) the onset of Case-B mass transfer as the
primary star fills its Roche lobe, (c) the system following the first supernova, and (d)
the final double compact object configuration. Stellar sizes and orbital separations are
drawn to scale relative to one another within each panel. Colors in the top row of panels
reflect the surface temperatures in TULIPS mode, while the bottom row uses cartoon style images. \label{fig:snapshots}](vimes.png)
Figure 1: Example snapshots from a VIMES animation of a massive binary system. From left to
right: (a) the initial main-sequence phase, (b) the onset of Case-B mass transfer as the
primary star fills its Roche lobe, (c) the system following the first supernova, and (d)
the final double compact object configuration. Stellar sizes and orbital separations are drawn to scale relative to one another within each panel. Colors of the stars in the top row of the panel are made using the TULIPS mode to reflect the effective surface temperatures, while the bottom row is made using the default cartoon images. Each column is a snapshot taken at the same evolutionary phase for the same system. 

# Dependencies
TULIPS, PyGame, ImageIO, and Pillow. 
VIMES is written in Python and builds on several open-source scientific Python packages. Numerical data processing — including array manipulation, linear interpolation of intermediate frames, and storage of processed frame data in compressed .npz format — relies on NumPy [@Harris:2020]. Reading COMPAS detailed output files, which are stored in HDF5 format, is handled by h5py [@Collette:2023]. The animation rendering pipeline uses PyGame to render frames and merge them into an MP4 file. [@Shinners:2025] The optional TULIPS color mode uses the temperature-to-color conversion utilities from the TULIPS package [@Laplace:2022]. Image assets used in the default cartoon rendering mode are loaded using Pillow [@Clark:2015].

# Acknowledgements
FSB 


# References
