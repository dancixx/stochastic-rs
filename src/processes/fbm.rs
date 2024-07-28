//! Fractional Brownian Motion (fBM) generator.

use crate::{noises::fgn::FgnFft, utils::Generator};
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;

/// Struct for generating Fractional Brownian Motion (fBM).
pub struct Fbm {
  #[allow(dead_code)]
  hurst: f64,
  #[allow(dead_code)]
  n: usize,
  m: Option<usize>,
  fgn: Option<FgnFft>,
}

impl Fbm {
  /// Creates a new `Fbm` instance.
  ///
  /// # Parameters
  ///
  /// - `hurst`: Hurst parameter, must be between 0 and 1.
  /// - `n`: Number of time steps.
  /// - `t`: Total time (optional, defaults to 1.0).
  /// - `m`: Number of parallel samples to generate (optional).
  ///
  /// # Returns
  ///
  /// A new `Fbm` instance.
  ///
  /// # Panics
  ///
  /// Panics if the `hurst` parameter is not between 0 and 1.
  ///
  /// # Example
  ///
  /// ```
  /// let fbm = Fbm::new(0.75, 1000, Some(1.0), Some(10));
  /// ```
  ///
  pub fn new(hurst: f64, n: usize, t: Option<f64>, m: Option<usize>) -> Self {
    if !(0.0..=1.0).contains(&hurst) {
      panic!("Hurst parameter must be in (0, 1)")
    }

    Self {
      hurst,
      n,
      m,
      fgn: Some(FgnFft::new(hurst, n - 1, t, None)),
    }
  }
}

impl Generator for Fbm {
  /// Generates a sample of fractional Brownian motion (fBM).
  ///
  /// # Returns
  ///
  /// A `Array1<f64>` representing the generated fBM sample.
  ///
  /// # Example
  ///
  /// ```
  /// let fbm = Fbm::new(0.75, 1000, Some(1.0), None);
  /// let sample = fbm.sample();
  /// ```
  fn sample(&self) -> Array1<f64> {
    let fgn = self.fgn.as_ref().unwrap().sample();
    let mut fbm = Array1::<f64>::from(fgn);
    fbm.accumulate_axis_inplace(Axis(0), |&x, y| *y += x);
    vec![0.0].into_iter().chain(fbm).collect()
  }

  /// Generates parallel samples of fractional Brownian motion (fBM).
  ///
  /// # Returns
  ///
  /// A `Array2<f64>>` representing the generated parallel fBM samples.
  ///
  /// # Panics
  ///
  /// Panics if `m` is not specified.
  ///
  /// # Example
  ///
  /// ```
  /// let fbm = Fbm::new(0.75, 1000, Some(1.0), Some(10));
  /// let samples = fbm.sample_par();
  /// ```
  fn sample_par(&self) -> Array2<f64> {
    if self.m.is_none() {
      panic!("Number of paths must be specified")
    }

    let mut xs = Array2::zeros((self.m.unwrap(), self.n));

    xs.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut x| {
      x.assign(&self.sample());
    });

    xs
  }
}
