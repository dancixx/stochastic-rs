use crate::utils::Generator;
use nalgebra::{DMatrix, DVector, Dim, Dyn, RowDVector};
use ndarray::{concatenate, prelude::*};
use ndarray_rand::RandomExt;
use ndrustfft::{ndfft_par, FftHandler};
use num_complex::{Complex, ComplexDistribution};
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::cmp::Ordering::{Equal, Greater, Less};

pub struct FgnFft {
  hurst: f64,
  n: usize,
  t: f64,
  sqrt_eigenvalues: Array1<Complex<f64>>,
  m: Option<usize>,
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
          * ((x as f64 + 1.0).powf(2.0 * hurst) - 2.0 * (x as f64).powf(2.0 * hurst)
            + (x as f64 - 1.0).powf(2.0 * hurst))
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
    }
  }
}

impl Generator for FgnFft {
  fn sample(&self) -> Vec<f64> {
    let rnd = Array1::<Complex<f64>>::random(
      2 * self.n,
      ComplexDistribution::new(StandardNormal, StandardNormal),
    );
    let mut fgn_fft_handler = FftHandler::new(2 * self.n);
    let mut fgn_fft = Array1::<Complex<f64>>::zeros(2 * self.n);
    ndfft_par(
      &(&self.sqrt_eigenvalues * &rnd),
      &mut fgn_fft,
      &mut fgn_fft_handler,
      0,
    );
    let fgn = fgn_fft
      .slice(s![1..self.n + 1])
      .mapv(|x| (x.re * (self.n as f64).powf(-self.hurst)) * self.t.powf(self.hurst));
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

pub struct FgnCholesky {
  hurst: f64,
  n: usize,
  t: Option<f64>,
  afc_sqrt: DMatrix<f64>,
  m: Option<usize>,
}

impl FgnCholesky {
  pub fn new(hurst: f64, n: usize, t: Option<f64>, m: Option<usize>) -> Self {
    if !(0.0..1.0).contains(&hurst) {
      panic!("Hurst parameter must be between 0 and 1");
    }

    let afc_matrix_sqrt = |n: usize, hurst: f64| -> DMatrix<f64> {
      let mut acf_v = RowDVector::<f64>::zeros(n);
      acf_v[0] = 1.0;

      for i in 1..n {
        let idx = i as f64;

        acf_v[i] = 0.5
          * ((idx + 1.0).powf(2.0 * hurst) - 2.0 * idx.powf(2.0 * hurst)
            + (idx - 1.0).powf(2.0 * hurst))
      }
      let mut m: nalgebra::Matrix<f64, Dyn, Dyn, nalgebra::VecStorage<f64, Dyn, Dyn>> =
        DMatrix::<f64>::zeros_generic(Dyn::from_usize(n), Dyn::from_usize(n));

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
    };

    Self {
      hurst,
      n,
      t,
      afc_sqrt: afc_matrix_sqrt(n, hurst),
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
