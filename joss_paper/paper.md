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
  - orcid: 0009-0009-5823-4399
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Floor S. Broekgaarden
  - orcid: 0000-0002-4421-4962
    affiliation: 1
  - name: Amedeo Romagnolo
  - orcid: 0000-0001-9583-4339
    affiliation: "2, 3"
  - name: Thomas Reichardt
  - orcid: 0000-0003-4630-3384
    affiliation: 4

affiliations:
 - name: Department of Astronomy and Astrophysics, University of California San Diego, 9500 Gilman Drive, La Jolla, CA 92093, USA
   index: 1
 - name: Universit\"at Heidelberg, Zentrum f\"ur Astronomie (ZAH), Institut f\"ur Theoretische Astrophysik, Albert Ueberle Str. 2, 69120, Heidelberg, Germany
   index: 2
 - name: Dipartimento di Fisica e Astronomia Galileo Galilei, Università di Padova, Vicolo dell’Osservatorio 3, I–35122 Padova, Italy
   index: 3
 - name: Astronomy Data and Computing Services (ADACS); the Centre for Astrophysics & Supercomputing, Swinburne University of Technology, P.O. Box 218, Hawthorn, VIC 3122, Australia
   index: 4

date: 12 June 2026
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/1538-4365/ac416c
aas-journal: Astrophysical Journal Supplements 
---

# Summary

Binary stellar evolution describes the complex life of two gravitationally bound stars, whose initial conditions and binary interactions (e.g., stable mass transfer and common-envelope episodes) ultimately determine the system's outcome and whether it produces specific systems such as compact object mergers detectable through gravitational waves. 
Rapid binary population synthesis codes enable large-scale simulations of these systems. They can calculate and track detailed time-series data for the evolving stellar and orbital properties throughout the binary’s lifetime. While these simulations are indispensable for theoretical studies, their detailed time-series numerical outputs can be difficult to interpret intuitively or communicate to a broader scientific audience because they are highly multidimensional and span many orders of magnitude across evolutionary timescales.

`VIMES` (VIsualization of Massive Evolving Stars) is a Python package that converts detailed output files into intuitive animations of binary stellar evolution. ach animation visualizes the evolving stellar radii, binary separation, effective temperatures (through color), eccentricity, and evolutionary phases of the binary components at every timestep.   `VIMES` is designed to handle the full range of binary configurations produced by rapid population-synthesis codes such as COMPAS and generates physically motivated 3D animations of the systems throughout their evolution. 
By bridging the gap between raw simulation data and intuitive visual representations, `VIMES` makes binary stellar evolution more accessible for scientific presentations, publications, teaching, and public outreach.


# Statement of need
Rapid population synthesis codes such as COMPAS [@Riley:2022, @TeamCompas:2025], COSMIC [@Breivik:2020], and SEVN [@Iorio:2023] are the primary tools for studying the evolution of binary stars and their formation pathways to exotic transients such as compact binary systems and their gravitational wave signatures [@Mandel:2022]. These codes produce data files containing numerical values for hundreds of physical quantities across thousands of timesteps. However, the diversity of binary evolutionary channels poses a persistent challenge for both analysis and outreach of the population synthesis simulations.

Although standard static diagrams:  HR diagrams, Kippenhahn diagrams, and binary evolution cartoon figures are widely used to summarize and visualize the outcomes of population synthesis simulations, they are sometimes challenging to infer, they are often hand-drawn schematic cartoons, and they do often not capture the full continuous evolution of a binary system but instead illustrate only the key evolutionary phases of a binary in a static and qualitative way.

`VIMES` addresses this gap. It is designed for researchers who study binary stellar evolution and need a quick, accurate way to visualize the evolution of a specific simulated system — whether to build intuition during analysis or to communicate results in
presentations and outreach. The animations produced are quantitatively accurate to the underlying simulation: all spatial scales, color temperatures, and evolutionary phases correspond directly to the data, rather than being schematic approximations.
`VIMES` allows for any binary system evolved with COMPAS or similar simulation codes to be turned into an animation, with the user having a choice over the type of images used for the visualization, as well as the type of scaling used when converting the data into an animation. 

 In doing so, `VIMES`  broadens access to binary evolution codes for researchers building physical intuition, for educators introducing stellar astrophysics, and for the public engaging with the science behind gravitational-wave sources. As population synthesis simulations continue to grow in scale and complexity, tools like `VIMES` will play an increasingly important role in making their outputs comprehensible and their insights widely shareable.

# State of the field

 Creating an accurate figure for a specific simulated binary is time-consuming and often requires omitting the evolutionary transitions between phases that are often scientifically important. Recently, [@2025ApJS..276...16W] created cogsworth which includes a pipeline to create static cartoons in 2D from population synthesis output, but an open-source tool that visualizes this as a continuous evolution, and that automates the creation of accurate, system-specific binary evolution from population synthesis code is still lacking. 

 `VIMES` was created to ensure that an effective way to visualize continuous evolution is possible while remaining accurate to the specific systems being simulated, rather than a generalized one.  
 
# Software Design 
`VIMES` processes detailed output files from population synthesis codes such as COMPAS in two stages: data ingestion and animation rendering.

## Data Ingestion and Frame Construction
`VIMES` reads the binary's evolution details (such as radius, effective temparature, mass, stellar type, mass transfer episodes, and time) from a COMPAS HDF5 detailed output file, extracting time-series data for both stars and the orbit. The data are first segmented into
evolutionary phases, defined by changes in the stellar type of either star (using the stellar type classification scheme of @Hurley:2000) or the onset of mass transfer. This
phase structure is used to determine frame sampling: rather than allocating frames proportional to time spent in each phase (which would result in animations dominated
by long, visually uneventful main-sequence phases), `VIMES` samples a fixed number of frames from each phase. This ensures that short but physically important phases — such
as common-envelope episodes or supernova kicks — receive adequate representation in the animation.
Where adjacent timesteps show large fractional or absolute changes in any displayed quantity, `VIMES` inserts linearly interpolated intermediate frames to ensure smooth visual transitions. The processed frames are cached as a compressed .npz file, decoupling data preparation from rendering and allowing the animation to be regenerated with different visual settings without re-reading the population synthesis output.


## Animation Rendering
The second stage reads the cached frame data and renders each frame as a two-dimensional representation of the binary system, with the two stars drawn at their correct relative sizes (proportional to stellar radius) and the correct orbital separation and eccentricity.
The orbit is drawn as an ellipse with the correct eccentricity, and the positions of the stars along the orbit are updated frame by frame. `VIMES` supports two visual modes:

 - Default mode
renders stars using a set of cartoon-style images that change
discretely with stellar type, making phase transitions immediately legible to an
audience unfamiliar with the underlying data conventions.
 - TULIPS mode
maps the effective surface temperature of each star to a
physically motivated RGB color using the temperature-to-color conversion from
the TULIPS package [@Laplace:2022], producing a more quantitatively accurate
visual at the cost of reduced contrast between evolutionary phases.

The rendered frames are assembled into a video file using matplotlib's animation framework [@Hunter:2007]. Figure \autoref{fig:snapshots} shows example snapshots from an animation of a double compact object progenitor system at four key evolutionary phases. VIMES works for COMPAS output but can easily be run to other population synthesis simulation outputs by changing the datafile to a similar datastructure or pointing VIMES to the required parameters. 



![Example snapshots from a VIMES animation of a massive binary system. From left to
right: (a) the initial main-sequence phase, (b) the onset of Case-B mass transfer as the
primary star fills its Roche lobe, (c) the system following the first supernova, and (d)
the final double compact object configuration. Stellar sizes and orbital separations are drawn to scale relative to one another within each panel. Colors of the stars in the top row of the panel are made using the TULIPS mode to reflect the effective surface temperatures, while the bottom row is made using the default cartoon images. Each column is a snapshot taken at the same evolutionary phase for the same system. \label{fig:snapshots}](vimes.png)


Together, these features make VIMES a versatile tool for navigating the complex binary pathways that population synthesis simulations produce — rendering the full diversity of evolutionary outcomes accessible to researchers, students, and the public, and ensuring that the rich information encoded in binary evolution codes can be fully explored and communicated.

# Research Impact Statement

`VIMES` provides a new way to interact with and communicate binary population-synthesis simulations by transforming detailed numerical outputs into physically motivated animations of stellar evolution. We expect the software to be broadly useful across gravitational-wave astrophysics, stellar evolution, and astronomy education, particularly as population-synthesis studies continue to grow in scale and complexity.

The ability to generate accurate, system-specific visualizations of binary evolution has applications ranging from scientific presentations and publications to classroom instruction and public outreach. In particular, `VIMES` enables researchers to more easily communicate complex evolutionary pathways,  including mass transfer, common-envelope evolution, supernovae, and compact-object formation,  to audiences that may not be familiar with the underlying simulation data structures.

Because the package is open source and modular, it can also be extended to additional population-synthesis frameworks and adapted for future visualization tools and educational platforms. We anticipate that tools such as `VIMES` will become increasingly valuable for improving the interpretability, accessibility, and communication of binary stellar-evolution simulations in the era of large-scale gravitational-wave astronomy.

`VIMES` is currently implemented for the `COMPAS` population-synthesis framework, with ongoing development to extend support to additional binary evolution codes including `COSMIC`, `SEVN`, and `StarTrack`. `VIMES` will also be integrated into the public-facing [GWLandscape](https://gwlandscape.org.au/single-binary-form/) platform, where users can evolve customizable binary systems and directly generate physically motivated 3D animations of their evolution without needing to install or run any local software. In addition, example animations and interactive visualizations are publicly available at the [GW Paleontology Lab Website](https://gwlandscape.org.au/single-binary-form/](https://floorbroekgaarden.github.io/interactive-figures/).


# Dependencies
TULIPS, PyGame, ImageIO, and Pillow. 
VIMES is written in Python and builds on several open-source scientific Python packages. Numerical data processing — including array manipulation, linear interpolation of intermediate frames, and storage of processed frame data in compressed .npz format — relies on NumPy [@Harris:2020]. Reading COMPAS detailed output files, which are stored in HDF5 format, is handled by h5py [@Collette:2023]. The animation rendering pipeline uses PyGame to render frames and merge them into an MP4 file. [@Shinners:2025] The optional TULIPS color mode uses the temperature-to-color conversion utilities from the TULIPS package [@Laplace:2022]. Image assets used in the default cartoon rendering mode are loaded using Pillow [@Clark:2015].

# AI usage disclosure

Generative AI was used in a limited role to help with debugging during the initial stages of this software. No generative AI was used in the creation or proofreading of this manuscript. All AI-suggested edits to the code were reviewed and confirmed by the authors, and all software design decisions and implementations were made by the authors. The AI tool used for this assistance was GitHub Copilot through VSCode. 

# Acknowledgements
The authors would like to thank all the members of the UCSD Gravitational-Wave Paleontology Lab for their constructive feedback and help with this project. We would like to thank the UCSD URS program for their support and for sponsoring this summer research project. 


# References
