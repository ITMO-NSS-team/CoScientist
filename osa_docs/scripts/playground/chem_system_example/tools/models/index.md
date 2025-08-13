
# Models
## Overview

The `models` module provides implementations for generative models, specifically a Conditional Variational Autoencoder (CVAE), along with associated utilities for data handling and inference. It facilitates the creation and use of models capable of generating data conditioned on class labels. This module contains functionalities for creating datasets, training the CVAE model via encoder and decoder, and subsequently generating new samples.

## Purpose

This module serves the purpose of generating synthetic data, tailored to specific categories or shapes, serving as a component for the broader CoScientist project. Specifically, the ability to generate diverse data samples could support the enrichment of the scientific literature database, enhance information retrieval, and provide a means for creating examples to train or evaluate other models within the project. The generated data can be used as a proxy or to augment limited scientific data in the chemistry domain, contributing to improved research support.