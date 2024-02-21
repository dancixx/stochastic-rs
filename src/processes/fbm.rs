use crate::{noises::fgn::FgnFft, utils::Generator};
use ndarray::{Array1, Axis};
use rayon::prelude::*;

pub struct Fbm {
  #[allow(dead_code)]
  hurst: f64,
  #[allow(dead_code)]
  n: usize,
  m: Option<usize>,
  fgn: Option<FgnFft>,
}

impl Fbm {
  pub fn new(hurst: f64, n: usize, t: Option<f64>, m: Option<usize>) -> Self {
    if !(0.0..1.0).contains(&hurst) {
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
  fn sample(&self) -> Vec<f64> {
    let fgn = self.fgn.as_ref().unwrap().sample();
    let mut fbm = Array1::<f64>::from_vec(fgn);
    fbm.accumulate_axis_inplace(Axis(0), |&x, y| *y += x);
    vec![0.0].into_iter().chain(fbm).collect()
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
