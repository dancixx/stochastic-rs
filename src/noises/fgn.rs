use crate::utils::Generator;
use ndarray::parallel::prelude::*;
use ndarray::{concatenate, prelude::*};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use ndrustfft::{ndfft_par, FftHandler};
use num_complex::{Complex, ComplexDistribution};

pub struct FgnFft {
  hurst: f64,
  n: usize,
  t: f64,
  sqrt_eigenvalues: Array1<Complex<f64>>,
  m: Option<usize>,
  fft_handler: FftHandler<f64>,
  fft_fgn: Array1<Complex<f64>>,
}

impl FgnFft {
  pub fn new(hurst: f64, n: usize, t: Option<f64>, m: Option<usize>) -> Self {
    if !(0.0..=1.0).contains(&hurst) {
      panic!("Hurst parameter must be between 0 and 1");
    }
    let mut r = Array1::linspace(0.0, n as f64, n + 1);
    r.par_mapv_inplace(|x| {
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
    let mut data = Array1::<Complex<f64>>::zeros(r.len());
    for (i, v) in r.iter().enumerate() {
      data[i] = Complex::new(*v, 0.0);
    }
    let mut r_fft = FftHandler::new(r.len());
    let mut sqrt_eigenvalues = Array1::<Complex<f64>>::zeros(r.len());
    ndfft_par(&data, &mut sqrt_eigenvalues, &mut r_fft, 0);
    sqrt_eigenvalues.par_mapv_inplace(|x| Complex::new((x.re / (2.0 * n as f64)).sqrt(), x.im));

    Self {
      hurst,
      n,
      t: t.unwrap_or(1.0),
      sqrt_eigenvalues,
      m,
      fft_handler: FftHandler::new(2 * n),
      fft_fgn: Array1::<Complex<f64>>::zeros(2 * n),
    }
  }
}

impl Generator for FgnFft {
  fn sample(&self) -> Vec<f64> {
    let rnd = Array1::<Complex<f64>>::random(
      2 * self.n,
      ComplexDistribution::new(StandardNormal, StandardNormal),
    );
    let fgn = &self.sqrt_eigenvalues * &rnd;
    let mut fft_handler = self.fft_handler.clone();
    let mut fgn_fft = self.fft_fgn.clone();
    ndfft_par(&fgn, &mut fgn_fft, &mut fft_handler, 0);
    let fgn = fgn_fft
      .slice(s![1..self.n + 1])
      .mapv(|x: Complex<f64>| (x.re * (self.n as f64).powf(-self.hurst)) * self.t.powf(self.hurst));
    fgn.to_vec()
  }

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
