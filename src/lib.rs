#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

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
//! ## License
//!
//! This project is licensed under the MIT License.
//!
//! ## Acknowledgements
//!
//! - Developed by [dancixx](https://github.com/dancixx).
//! - Contributions and feedback are welcome!

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

pub mod ai;
pub mod quant;
pub mod stats;
pub mod stochastic;
