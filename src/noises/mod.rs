//! This module contains the implementations of various noise generation processes.
//!
//! The following noise generation processes are implemented:
//!
//! - **Fractional Gaussian Noise (FGN)**
//!   - Generates fractional Gaussian noise using the Fast Fourier Transform (FFT) approach.
//!
//! - **Gaussian Noise (GN)**
//!   - Generates Gaussian noise, commonly used in simulations requiring white noise or random perturbations.

pub mod fgn;
pub mod gn;
