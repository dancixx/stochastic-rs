use crate::utils::Generator;
use nalgebra::{DMatrix, DVector, Dim, Dyn, RowDVector};
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::cmp::Ordering::{Equal, Greater, Less};

pub struct FgnCholesky {
  hurst: f64,
  n: usize,
  t: Option<f64>,
  afc_sqrt: DMatrix<f64>,
  m: Option<usize>,
}

impl FgnCholesky {
  pub fn new(hurst: f64, n: usize, t: Option<f64>, m: Option<usize>) -> Self {
    if !(0.0..=1.0).contains(&hurst) {
      panic!("Hurst parameter must be between 0 and 1");
    }
    let afc_sqrt = afc_matrix_sqrt(n, hurst);

    Self {
      hurst,
      n,
      t,
      afc_sqrt,
      m,
    }
  }
}

impl Generator for FgnCholesky {
  fn sample(&self) -> Vec<f64> {
    let noise = thread_rng()
      .sample_iter::<f64, StandardNormal>(StandardNormal)
      .take(self.n)
      .collect();
    let noise = DVector::<f64>::from_vec(noise);

    ((self.afc_sqrt.clone() * noise).transpose()
      * (self.n as f64).powf(-self.hurst)
      * self.t.unwrap_or(1.0).powf(self.hurst))
    .data
    .as_vec()
    .clone()
  }

  fn sample_par(&self) -> Vec<Vec<f64>> {
    if self.m.is_none() {
      panic!("m must be specified for parallel sampling")
    }
    (0..self.m.unwrap())
      .into_par_iter()
      .map(|_| self.sample())
      .collect()
  }
}

pub fn fgn(hurst: f64, n: usize, t: Option<f64>) -> Vec<f64> {
  if !(0.0..1.0).contains(&hurst) {
    panic!("Hurst parameter must be in (0, 1)")
  }

  let acf_sqrt = afc_matrix_sqrt(n, hurst);
  let noise = thread_rng()
    .sample_iter::<f64, StandardNormal>(StandardNormal)
    .take(n)
    .collect();
  let noise = DVector::<f64>::from_vec(noise);

  ((acf_sqrt * noise).transpose() * (n as f64).powf(-hurst) * t.unwrap_or(1.0).powf(hurst))
    .data
    .as_vec()
    .clone()
}

fn afc_vector(n: usize, hurst: f64) -> RowDVector<f64> {
  let mut v = RowDVector::<f64>::zeros(n);
  v[0] = 1.0;

  for i in 1..n {
    let idx = i as f64;

    v[i] = 0.5
      * ((idx + 1.0).powf(2.0 * hurst) - 2.0 * idx.powf(2.0 * hurst)
        + (idx - 1.0).powf(2.0 * hurst))
  }

  v
}

fn afc_matrix_sqrt(n: usize, hurst: f64) -> DMatrix<f64> {
  let acf_v = afc_vector(n, hurst);
  let mut m = DMatrix::<f64>::zeros_generic(Dyn::from_usize(n), Dyn::from_usize(n));

  for i in 0..n {
    for j in 0..n {
      match i.cmp(&j) {
        Equal => m[(i, j)] = acf_v[0],
        Greater => m[(i, j)] = acf_v[i - j],
        Less => continue,
      }
    }
  }

  m.cholesky().unwrap().l()
}
