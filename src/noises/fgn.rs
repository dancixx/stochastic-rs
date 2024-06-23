//! Fractional Gaussian Noise (FGN) generator using FFT.
//!
//! The `FgnFft` struct provides methods to generate fractional Gaussian noise (FGN)
//! using the Fast Fourier Transform (FFT) approach.

use crate::utils::Generator;
use ndarray::parallel::prelude::*;
use ndarray::{concatenate, prelude::*};
use ndarray_rand::rand_distr::StandardNormal;
use ndrustfft::{ndfft_par, FftHandler};
use num_complex::Complex;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::Distribution;
use rayon::slice::ParallelSliceMut;

/// Struct for generating Fractional Gaussian Noise (FGN) using FFT.
pub struct FgnFft {
  hurst: f64,
  n: usize,
  offset: usize,
  t: f64,
  sqrt_eigenvalues: Array1<Complex<f64>>,
  m: Option<usize>,
  fft_handler: FftHandler<f64>,
  fft_fgn: Array1<Complex<f64>>,
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
  /// use stochastic_rs::noises::fgn::FgnFft;
  ///
  /// let fgn_fft = FgnFft::new(0.75, 1000, Some(1.0), Some(10));
  /// ```
  pub fn new(hurst: f64, n: usize, t: Option<f64>, m: Option<usize>) -> Self {
    if !(0.0..=1.0).contains(&hurst) {
      panic!("Hurst parameter must be between 0 and 1");
    }

    let offset = n.next_power_of_two() - n;
    let n = n.next_power_of_two();
    let mut r = Array1::from_shape_fn(n + 1, |i| {
      let x = i as f64;
      let is_zero = (i == 0) as i32 as f64;
      let is_non_zero = 1.0 - is_zero;
      let non_zero_value = 0.5
        * ((x + 1.0).powf(2.0 * hurst) - 2.0 * x.powf(2.0 * hurst) + (x - 1.0).powf(2.0 * hurst));
      is_zero * 1.0 + is_non_zero * non_zero_value
    });
    r[0] = 1.0;

    let r = concatenate(
      Axis(0),
      #[allow(clippy::reversed_empty_ranges)]
      &[r.view(), r.slice(s![..;-1]).slice(s![1..-1]).view()],
    )
    .unwrap();
    let data = r.mapv(|v| Complex::new(v, 0.0));
    let mut r_fft = FftHandler::new(r.len());
    let mut sqrt_eigenvalues = Array1::<Complex<f64>>::zeros(r.len());
    ndfft_par(&data, &mut sqrt_eigenvalues, &mut r_fft, 0);
    sqrt_eigenvalues.par_mapv_inplace(|x| Complex::new((x.re / (2.0 * n as f64)).sqrt(), x.im));

    Self {
      hurst,
      n,
      offset,
      t: t.unwrap_or(1.0),
      sqrt_eigenvalues,
      m,
      fft_handler: FftHandler::new(2 * n),
      fft_fgn: Array1::<Complex<f64>>::zeros(2 * n),
    }
  }
}

impl Generator for FgnFft {
  /// Generates a sample of fractional Gaussian noise (FGN).
  ///
  /// # Returns
  ///
  /// A `Vec<f64>` representing the generated FGN sample.
  ///
  /// # Example
  ///
  /// ```
  /// use stochastic_rs::noises::fgn::FgnFft;
  ///
  /// let fgn_fft = FgnFft::new(0.75, 1000, Some(1.0), None);
  /// let sample = fgn_fft.sample();
  /// ```
  fn sample(&self) -> Vec<f64> {
    let num_threads = rayon::current_num_threads();
    let rngs = (0..num_threads)
      .map(|_| SmallRng::from_entropy())
      .collect::<Vec<SmallRng>>();
    let mut rnd = Array1::zeros(2 * self.n);
    let chunk_size = (2 * self.n + num_threads - 1) / num_threads;
    rnd
      .as_slice_mut()
      .expect("Failed to convert Array1 to slice")
      .par_chunks_mut(chunk_size)
      .zip(rngs.into_par_iter())
      .for_each(|(chunk, mut rng)| {
        for x in chunk.iter_mut() {
          let real: f64 = StandardNormal.sample(&mut rng);
          let img: f64 = StandardNormal.sample(&mut rng);
          *x = Complex::new(real, img);
        }
      });

    let fgn = &self.sqrt_eigenvalues * &rnd;
    let mut fft_handler = self.fft_handler.clone();
    let mut fgn_fft = self.fft_fgn.clone();
    ndfft_par(&fgn, &mut fgn_fft, &mut fft_handler, 0);
    let fgn = fgn_fft
      .slice(s![1..(self.n - self.offset) + 1])
      .mapv(|x: Complex<f64>| (x.re * (self.n as f64).powf(-self.hurst)) * self.t.powf(self.hurst));
    fgn.to_vec()
  }

  /// Generates parallel samples of fractional Gaussian noise (FGN).
  ///
  /// # Returns
  ///
  /// A `Vec<Vec<f64>>` representing the generated parallel FGN samples.
  ///
  /// # Panics
  ///
  /// Panics if `m` is not specified.
  ///
  /// # Example
  ///
  /// ```
  /// use stochastic_rs::noises::fgn::FgnFft;
  ///
  /// let fgn_fft = FgnFft::new(0.75, 1000, Some(1.0), Some(10));
  /// let samples = fgn_fft.sample_par();
  /// ```
  fn sample_par(&self) -> Vec<Vec<f64>> {
    if self.m.is_none() {
      panic!("m must be specified for parallel sampling");
    }

    (0..self.m.unwrap())
      .into_par_iter()
      .map(|_| self.sample())
      .collect()
  }
}
