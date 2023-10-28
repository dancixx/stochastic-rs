use crate::{
  noises::fgn::{FgnCholesky, FgnFft},
  utils::{Generator, NoiseGenerationMethod},
};
use ndarray::{Array1, Axis};
use rayon::prelude::*;

pub struct Fbm {
  #[allow(dead_code)]
  hurst: f64,
  #[allow(dead_code)]
  n: usize,
  m: Option<usize>,
  method: NoiseGenerationMethod,
  fgn_fft: Option<FgnFft>,
  fgn_cholesky: Option<FgnCholesky>,
}

impl Fbm {
  pub fn new(
    hurst: f64,
    n: usize,
    t: Option<f64>,
    m: Option<usize>,
    method: Option<NoiseGenerationMethod>,
  ) -> Self {
    if !(0.0..1.0).contains(&hurst) {
      panic!("Hurst parameter must be in (0, 1)")
    }

    match method.unwrap_or(NoiseGenerationMethod::Fft) {
      NoiseGenerationMethod::Fft => Self {
        hurst,
        n,
        m,
        method: NoiseGenerationMethod::Fft,
        fgn_fft: Some(FgnFft::new(hurst, n - 1, t, None)),
        fgn_cholesky: None,
      },
      NoiseGenerationMethod::Cholesky => Self {
        hurst,
        n,
        m,
        method: NoiseGenerationMethod::Cholesky,
        fgn_fft: None,
        fgn_cholesky: Some(FgnCholesky::new(hurst, n - 1, t, None)),
      },
    }
  }
}

impl Generator for Fbm {
  fn sample(&self) -> Vec<f64> {
    let fgn = match self.method {
      NoiseGenerationMethod::Fft => self.fgn_fft.as_ref().unwrap().sample(),
      NoiseGenerationMethod::Cholesky => self.fgn_cholesky.as_ref().unwrap().sample(),
    };
    let mut fbm = Array1::<f64>::from_vec(fgn);
    fbm.accumulate_axis_inplace(Axis(0), |&x, y| *y += x);
    vec![0.0].into_iter().chain(fbm.into_iter()).collect()
  }

  fn sample_par(&self) -> Vec<Vec<f64>> {
    if self.m.is_none() {
      panic!("Number of paths must be specified")
    }

    (0..self.m.unwrap())
      .into_par_iter()
      .map(|_| self.sample())
      .collect()
  }
}
