pub mod diffusion;
pub mod interest;
pub mod jump;
pub mod malliavin;
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
  /// Sample the process
  fn sample(&self) -> Array1<T>;

  /// Parallel sampling
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

  /// Number of time steps
  fn n(&self) -> usize;

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize>;

  /// Distribution of the process
  fn distribution(&mut self) {}

  /// Malliavin derivative of the process
  fn malliavin(&self) -> Array1<T> {
    unimplemented!()
  }
}

pub trait Sampling2D<T: Clone + Send + Sync + Zero>: Send + Sync {
  /// Sample the process
  fn sample(&self) -> [Array1<T>; 2];

  /// Parallel sampling
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

  /// Number of time steps
  fn n(&self) -> usize;

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize>;

  /// Malliavin derivative of the process
  fn malliavin(&self) -> [Array1<T>; 2] {
    unimplemented!()
  }
}

pub trait Sampling3D<T: Clone + Send + Sync + Zero>: Send + Sync {
  /// Sample the process
  fn sample(&self) -> [Array1<T>; 3];

  /// Parallel sampling
  fn sample_par(&self) -> [Array2<T>; 3] {
    unimplemented!()
  }

  /// Number of time steps
  fn n(&self) -> usize;

  /// Number of samples for parallel sampling
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
