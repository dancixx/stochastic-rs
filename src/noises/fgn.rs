//! Fractional Gaussian Noise (FGN) generator using FFT.
//!
//! The `FgnFft` struct provides methods to generate fractional Gaussian noise (FGN)
//! using the Fast Fourier Transform (FFT) approach.

use std::sync::{Arc, RwLock};

use crate::utils::Generator;
use ndarray::parallel::prelude::*;
use ndarray::{concatenate, prelude::*};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use ndrustfft::{ndfft, FftHandler};
use num_complex::{Complex, ComplexDistribution};

/// Struct for generating Fractional Gaussian Noise (FGN) using FFT.
pub struct FgnFft {
  hurst: f64,
  n: usize,
  offset: usize,
  t: f64,
  sqrt_eigenvalues: Arc<Array1<Complex<f64>>>,
  m: Option<usize>,
  fft_handler: Arc<FftHandler<f64>>,
  fft_fgn: Arc<RwLock<Array1<Complex<f64>>>>,
}

impl FgnFft {
  /// Creates a new `FgnFft` instance.
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
  /// A new `FgnFft` instance.
  ///
  /// # Panics
  ///
  /// Panics if the `hurst` parameter is not between 0 and 1.
  ///
  /// # Example
  ///
  /// ```
  /// let fgn_fft = FgnFft::new(0.75, 1000, Some(1.0), Some(10));
  /// ```
  pub fn new(hurst: f64, n: usize, t: Option<f64>, m: Option<usize>) -> Self {
    if !(0.0..=1.0).contains(&hurst) {
      panic!("Hurst parameter must be between 0 and 1");
    }
    let n_ = n.next_power_of_two();
    let offset = n_ - n;
    let n = n_;
    let mut r = Array1::linspace(0.0, n as f64, n + 1);
    r.mapv_inplace(|x| {
      if x == 0.0 {
        1.0
      } else {
        0.5
          * ((x + 1.0).powf(2.0 * hurst) - 2.0 * x.powf(2.0 * hurst) + (x - 1.0).powf(2.0 * hurst))
      }
    });
    let r = concatenate(
      Axis(0),
      #[allow(clippy::reversed_empty_ranges)]
      &[r.view(), r.slice(s![..;-1]).slice(s![1..-1]).view()],
    )
    .unwrap();
    let data = r.mapv(|v| Complex::new(v, 0.0));
    let r_fft = FftHandler::new(r.len());
    let mut sqrt_eigenvalues = Array1::<Complex<f64>>::zeros(r.len());
    ndfft(&data, &mut sqrt_eigenvalues, &r_fft, 0);
    sqrt_eigenvalues.mapv_inplace(|x| Complex::new((x.re / (2.0 * n as f64)).sqrt(), x.im));

    Self {
      hurst,
      n,
      offset,
      t: t.unwrap_or(1.0),
      sqrt_eigenvalues: Arc::new(sqrt_eigenvalues),
      m,
      fft_handler: Arc::new(FftHandler::new(2 * n)),
      fft_fgn: Arc::new(RwLock::new(Array1::<Complex<f64>>::zeros(2 * n))),
    }
  }
}

impl Generator for FgnFft {
  /// Generates a sample of fractional Gaussian noise (FGN).
  ///
  /// # Returns
  ///
  /// A `Array1<f64>` representing the generated FGN sample.
  ///
  /// # Example
  ///
  /// ```
  /// let fgn_fft = FgnFft::new(0.75, 1000, Some(1.0), None);
  /// let sample = fgn_fft.sample();
  /// ```
  fn sample(&self) -> Array1<f64> {
    let rnd = Array1::<Complex<f64>>::random(
      2 * self.n,
      ComplexDistribution::new(StandardNormal, StandardNormal),
    );
    let fgn = &*self.sqrt_eigenvalues * &rnd;
    let mut fgn_fft = self.fft_fgn.read().unwrap().clone();
    ndfft(&fgn, &mut fgn_fft, &*self.fft_handler, 0);
    let scale = (self.n as f64).powf(-self.hurst) * self.t.powf(self.hurst);
    let fgn = fgn_fft
      .slice(s![1..self.n - self.offset + 1])
      .mapv(|x: Complex<f64>| x.re * scale);
    fgn
  }

  /// Generates parallel samples of fractional Gaussian noise (FGN).
  ///
  /// # Returns
  ///
  /// A `Array2<f64>>` representing the generated parallel FGN samples.
  ///
  /// # Panics
  ///
  /// Panics if `m` is not specified.
  ///
  /// # Example
  ///
  /// ```
  /// let fgn_fft = FgnFft::new(0.75, 1000, Some(1.0), Some(10));
  /// let samples = fgn_fft.sample_par();
  /// ```
  fn sample_par(&self) -> Array2<f64> {
    if self.m.is_none() {
      panic!("m must be specified for parallel sampling");
    }

    let mut xs = Array2::zeros((self.m.unwrap(), self.n - self.offset));

    xs.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut x| {
      x.assign(&self.sample().slice(s![..self.n - self.offset]));
    });

    xs
  }
}

#[cfg(test)]
mod tests {
  use plotly::{common::Line, Plot, Scatter};

  use super::*;

  #[test]
  fn plot() {
    let fgn = FgnFft::new(0.9, 1000, Some(1.0), Some(1));
    let mut plot = Plot::new();
    let d = fgn.sample_par();
    for data in d.axis_iter(Axis(0)) {
      let trace = Scatter::new((0..data.len()).collect::<Vec<_>>(), data.to_vec())
        .mode(plotly::common::Mode::Lines)
        .line(
          Line::new()
            .color("orange")
            .shape(plotly::common::LineShape::Linear),
        )
        .name("Fgn");
      plot.add_trace(trace);
    }
    plot.show();
  }
}
