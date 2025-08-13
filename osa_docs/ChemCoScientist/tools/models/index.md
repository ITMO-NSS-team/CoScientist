
# Models
## Overview
This module implements the core components for conditional image generation, specifically a Conditional Variational Autoencoder (CVAE). It includes functionalities for creating and manipulating datasets, encoding and decoding images, and performing inference to generate new samples.

## Purpose
This module provides the necessary tools to generate images conditioned on a specific shape. It’s used to create synthetic data to support the project’s goal of analyzing scientific papers and answering user questions. This is achieved through the CVAE model, which learns to generate images (likely representing molecular structures or related visualizations) based on input labels representing different shapes, enabling data augmentation or potentially aiding in the visualization of concepts described in scientific literature. The module facilitates functionalities like converting labels to a one-hot encoding, creating datasets, and performing complete encoding/decoding process.