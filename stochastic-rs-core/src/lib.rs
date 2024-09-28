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
use rand_distr::Distribution;

pub trait ProcessDistribution: Distribution<f64> + Copy + Send + Sync + Default {}

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
