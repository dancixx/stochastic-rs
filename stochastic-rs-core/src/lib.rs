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

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

pub mod diffusions;
pub mod interest;
pub mod jumps;
pub mod noises;
pub mod processes;
pub mod volatility;

use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Axis};
use ndrustfft::Zero;

pub trait Sampling<T: Clone + Send + Sync + Zero>: Send + Sync {
  fn sample(&self) -> Array1<T>;
  fn sample_par(&self) -> Array2<T> {
    if self.m().is_none() {
      panic!("m must be specified for parallel sampling");
    }

    let mut xs = Array2::zeros((self.m().unwrap(), self.n()));

    xs.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut x| {
      x.assign(&self.sample());
    });

    xs
  }
  fn n(&self) -> usize;
  fn m(&self) -> Option<usize>;
}

pub trait Sampling2D<T: Clone + Send + Sync + Zero>: Send + Sync {
  fn sample(&self) -> [Array1<T>; 2];
  fn sample_par(&self) -> [Array2<T>; 2] {
    unimplemented!()
  }
  fn n(&self) -> usize;
  fn m(&self) -> Option<usize>;
}

pub trait Sampling3D<T: Clone + Send + Sync + Zero>: Send + Sync {
  fn sample(&self) -> [Array1<T>; 3];
  fn sample_par(&self) -> [Array2<T>; 3] {
    unimplemented!()
  }
  fn n(&self) -> usize;
  fn m(&self) -> Option<usize>;
}
