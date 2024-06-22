//! # Stochastic Processes
//!
//! `stochastic-rs` is a Rust library that provides a comprehensive set of tools to simulate and analyze stochastic processes.
//! This library is useful for researchers, practitioners, and anyone interested in modeling and simulating various types of stochastic processes.
//!
//! ## Features
//!
//! - Simulation of various diffusion processes including Brownian motion, geometric Brownian motion, Ornstein-Uhlenbeck process, and more.
//! - Generation of noise processes such as Gaussian noise, fractional Gaussian noise, and Poisson processes.
//! - Modeling with advanced stochastic models like the Heston model and the Bates model.
//! - Calculation of fractal dimensions and other statistical properties of time series data.
//! - Support for jump processes and compound Poisson processes.
//!
//! ## Modules
//!
//! - [`prelude`]: Re-exports of common types and traits for easier usage.
//! - [`diffusions`]: Contains implementations of various diffusion processes.
//! - [`jumps`]: Contains implementations of jump processes.
//! - [`models`]: Contains implementations of advanced stochastic models.
//! - [`noises`]: Contains implementations of noise generation processes.
//! - [`processes`]: Contains implementations of various stochastic processes.
//! - [`statistics`]: Contains tools for statistical analysis of time series data.
//! - [`utils`]: Contains utility functions and helpers.
//!
//! ## Examples
//!
//! ```rust
//! use stochastic_rs::prelude::*;
//! use stochastic_rs::diffusions::bm;
//!
//! // Simulate a Brownian motion process
//! let n = 1000;
//! let t = 1.0;
//! let bm_path = bm(n, Some(t));
//! println!("Brownian Motion Path: {:?}", bm_path);
//! ```
//!
//! ```rust
//! use stochastic_rs::statistics::higuchi_fd;
//!
//! // Calculate the Higuchi Fractal Dimension of a time series
//! let time_series = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
//! let fd = higuchi_fd(&time_series, 5);
//! println!("Higuchi Fractal Dimension: {}", fd);
//! ```
//!
//! ## License
//!
//! This project is licensed under the MIT License.
//!
//! ## Acknowledgements
//!
//! - Developed by [dancixx](https://github.com/dancixx).
//! - Contributions and feedback are welcome!

pub mod prelude;

pub mod diffusions;
pub mod jumps;
pub mod models;
pub mod noises;
pub mod processes;
pub mod statistics;
pub mod utils;
