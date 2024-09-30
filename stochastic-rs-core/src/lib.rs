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

pub mod diffusion;
pub mod interest;
pub mod jump;
pub mod noise;
pub mod process;
pub mod volatility;

use std::sync::{Arc, Mutex};

use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Axis};
use ndrustfft::Zero;
use num_complex::Complex64;
use rand_distr::Distribution as RandDistribution;

pub trait ProcessDistribution: RandDistribution<f64> + Copy + Send + Sync + Default {}

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
  fn distribution(&mut self) {}
}

pub trait Sampling2D<T: Clone + Send + Sync + Zero>: Send + Sync {
  fn sample(&self) -> [Array1<T>; 2];
  fn sample_par(&self) -> [Array2<T>; 2] {
    if self.m().is_none() {
      panic!("m must be specified for parallel sampling");
    }

    let m = self.m().unwrap(); // m értékét előre kinyerjük, hogy ne kelljen többször unwrap-elni
    let xs1 = Arc::new(Mutex::new(Array2::zeros((self.m().unwrap(), self.n()))));
    let xs2 = Arc::new(Mutex::new(Array2::zeros((self.m().unwrap(), self.n()))));

    (0..m).into_par_iter().for_each(|i| {
      let [x1, x2] = self.sample(); // Minden szálon mintavételezünk
      xs1.lock().unwrap().row_mut(i).assign(&x1); // Az első mintavételezés eredményét beírjuk az első mátrix i. sorába
      xs2.lock().unwrap().row_mut(i).assign(&x2); // A második mintavételezés eredményét beírjuk a második mátrix i. sorába
    });

    let xs1 = xs1.lock().unwrap().clone();
    let xs2 = xs2.lock().unwrap().clone();
    [xs1, xs2]
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

pub trait Distribution {
  /// Characteristic function of the distribution
  fn characteristic_function(&self, _t: f64) -> Complex64 {
    Complex64::new(0.0, 0.0)
  }

  /// Probability density function of the distribution
  fn pdf(&self, _x: f64) -> f64 {
    0.0
  }

  /// Cumulative distribution function of the distribution
  fn cdf(&self, _x: f64) -> f64 {
    0.0
  }

  /// Inverse cumulative distribution function of the distribution
  fn inv_cdf(&self, _p: f64) -> f64 {
    0.0
  }

  /// Mean of the distribution
  fn mean(&self) -> f64 {
    0.0
  }

  /// Median of the distribution
  fn median(&self) -> f64 {
    0.0
  }

  /// Mode of the distribution
  fn mode(&self) -> f64 {
    0.0
  }

  /// Variance of the distribution
  fn variance(&self) -> f64 {
    0.0
  }

  /// Skewness of the distribution
  fn skewness(&self) -> f64 {
    0.0
  }

  /// Kurtosis of the distribution
  fn kurtosis(&self) -> f64 {
    0.0
  }

  /// Entropy of the distribution
  fn entropy(&self) -> f64 {
    0.0
  }

  /// Moment generating function of the distribution
  fn moment_generating_function(&self, _t: f64) -> f64 {
    0.0
  }
}
