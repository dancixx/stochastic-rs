use crate::utils::{FractionalNoiseGenerationMethod, Generator};
use nalgebra::{ComplexField, DMatrix, DVector, Dim, Dyn, RowDVector};
use ndarray::{concatenate, prelude::*};
use ndrustfft::{ndfft, ndfft_par, ndifft, FftHandler};
use num_complex::Complex;
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal, StandardNormal};
use rayon::prelude::*;
use std::{
  cmp::Ordering::{Equal, Greater, Less},
  sync::{Arc, Mutex},
};

pub struct FgnFft {
  hurst: f64,
  n: usize,
  t: f64,
  sqrt_eigenvalues: Array1<f64>,
  m: Option<usize>,
  method: FractionalNoiseGenerationMethod,
}

impl FgnFft {
  pub fn new(
    hurst: f64,
    n: usize,
    t: Option<f64>,
    m: Option<usize>,
    method: Option<FractionalNoiseGenerationMethod>,
  ) -> Self {
    if !(0.0..=1.0).contains(&hurst) {
      panic!("Hurst parameter must be between 0 and 1");
    }

    match method.unwrap_or(FractionalNoiseGenerationMethod::Kroese) {
      FractionalNoiseGenerationMethod::Kroese => {
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
          sqrt_eigenvalues: sqrt_eigenvalues.mapv(|x| x.re),
          m,
          method: method.unwrap_or(FractionalNoiseGenerationMethod::Kroese),
        }
      }
      FractionalNoiseGenerationMethod::DaviesHarte => {
        // let r = Array1::range(0.0, (n + 1) as f64, 1.0) * (2.0 * hurst);
        // let mut result = Array1::<f64>::zeros(n + 1);
        // result[0] = 1.0;
        // for i in 1..n {
        //   result[i] = 0.5 * (r[i - 1] - 2.0 * r[i] + r[i + 1]);
        // }

        // let mut data = Array1::<Complex<f64>>::zeros(result.len());
        // for (i, v) in result.iter().enumerate() {
        //   data[i] = Complex::new(*v, 0.0);
        // }
        // let mut r_fft = FftHandler::new(result.len());
        // let mut sqrt_eigenvalues = Array1::<Complex<f64>>::zeros(result.len());
        // ndifft(&data, &mut sqrt_eigenvalues, &mut r_fft, 0);
        // sqrt_eigenvalues.mapv_inplace(|x| x.powf(0.5));

        // Self {
        //   hurst,
        //   n,
        //   t: t.unwrap_or(1.0),
        //   sqrt_eigenvalues: sqrt_eigenvalues.mapv(|x| x.re),
        //   m,
        //   method: method.unwrap_or(FractionalNoiseGenerationMethod::Kroese),
        // }

        todo!("Davies-Harte method is not implemented yet")
      }
    }
  }
}

impl Generator for FgnFft {
  fn sample(&self) -> Vec<f64> {
    match self.method {
      FractionalNoiseGenerationMethod::Kroese => {
        let mut rnd = Array1::<Complex<f64>>::zeros(2 * self.n);
        rnd.par_mapv_inplace(|_| {
          Complex::new(
            rand::thread_rng().sample(StandardNormal),
            rand::thread_rng().sample(StandardNormal),
          )
        });
        let mut fgn = Array1::<Complex<f64>>::zeros(2 * self.n);
        for (i, v) in rnd.iter().enumerate() {
          fgn[i] = self.sqrt_eigenvalues[i] * v;
        }
        let mut fgn_fft_handler = FftHandler::new(2 * self.n);
        let mut fgn_fft = Array1::<Complex<f64>>::zeros(2 * self.n);
        ndfft_par(&fgn, &mut fgn_fft, &mut fgn_fft_handler, 0);

        let mut fgn: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> =
          fgn_fft.slice(s![1..self.n + 1]).mapv(|x| x.re);
        fgn.par_mapv_inplace(|x| (x * (self.n as f64).powf(-self.hurst)) * self.t.powf(self.hurst));
        fgn.to_vec()
      }
      FractionalNoiseGenerationMethod::DaviesHarte => {
        // let m = 2usize.pow((self.n - 2).next_power_of_two() as u32) + 1;
        // let scale = (self.t / self.n as f64).powf(self.hurst) * 2.0.powf(0.5) * (m - 1) as f64;
        // let mut rng = rand::thread_rng();
        // let normal = Normal::new(0.0, scale).unwrap();
        // let mut rnd = Array1::<Complex<f64>>::zeros(m);
        // for (_, v) in rnd.iter_mut().enumerate() {
        //   let real = normal.sample(&mut rng) * 2.0.powf(0.5);
        //   let imag = normal.sample(&mut rng) * 2.0.powf(0.5);
        //   *v = Complex::new(real, imag);
        // }

        // let mut _fgn = Array1::<Complex<f64>>::zeros(m);
        // for (i, v) in rnd.iter().enumerate() {
        //   _fgn[i] = self.sqrt_eigenvalues[i] * v;
        // }
        // let mut fgn_fft = FftHandler::new(m);
        // let mut fgn = Array1::<Complex<f64>>::zeros(m);
        // ndifft(&_fgn, &mut fgn, &mut fgn_fft, 0);

        // fgn.slice(s![0..self.n]).mapv(|x| x.re).to_vec()

        todo!("Davies-Harte method is not implemented yet")
      }
    }
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
