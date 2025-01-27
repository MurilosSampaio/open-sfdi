# Portable Spatial Frequency Domain Imaging (SFDI) Device

## Overview

This repository contains the open-source code developed for the [Portable Spatial Frequency Domain Imaging Device](https://github.com/MurilosSampaio/open-sfdi) as described in the article:

**Murilo S. Sampaio et al., "Portable Spatial Frequency Domain Imaging Device with Raspberry Pi and Open-Source Tools"**

The code enables the generation of graphs and analysis of tissue optical properties, specifically the absorption (μa) and reduced scattering (μs’) coefficients, using data acquired from the SFDI system. The system leverages a Raspberry Pi 4B, an HY300 LCD projector, and a Raspberry Pi OV5647 camera module to perform in-vivo measurements of tissue optical properties.

## Requirements

### Hardware

- **Raspberry Pi 4B** with 4 GB RAM
- **HY300 LCD Projector** with LED light
- **Raspberry Pi OV5647 Camera Module**
- **7-inch LCD Touchscreen Display**
- **3D-Printed Case and Filter Holder** (design files available [here](https://github.com/MurilosSampaio/open-sfdi/tree/main/3D_Print))
- **Thorlabs Bandpass Optical Filters** (450-680 nm range)
- **Other Components:** Acrylic base, threaded bars, epoxy resin for phantoms, etc.

### Software

- **Raspberry Pi OS**
- **Python 3.8+**
- **Picamera2 Python Module**
- **OpenCV**
- **NumPy**
- **SciPy**
- **Matplotlib**
- **Tkinter**
- **Additional Python Libraries:**
  - `scikit-learn` 
  - `pandas` 

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/MurilosSampaio/open-sfdi.git
   cd open-sfdi

   mkdir reference_images
   mkdir saved_pictures
   mkdir results
   
## Usage

Run the main.py file in a system with double screen (being one of the screns a projector), you must have a picamera installed in the raspberry.

The GUI is going to open from the main.py after loading the machine learning models. In the code you can change to calculate using the Rational Functions or the Machine Learning Models.

First you must capture the reference phantom values and then the samples images. 

(note, the picamera sometimes needs some distortion correction or calibration, the models use a 20 mm-1 spatial frequency, make sure to ajust the images to have the correct distance between the projections fringes)

### Acknowledgments

- Fundamentals: This project was financed by Fundação de Amparo à Pesquisa do Estado de S. Paulo – FAPESP, Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES), and the National Council for Scientific and Technological Development (CNPq).
- Special Thanks: Arthur Melo de Oliveira for assistance in modeling and printing the equipment structure.
