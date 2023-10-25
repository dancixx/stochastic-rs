use crate::{
  noises::fgn::{FgnCholesky, FgnFft},
  utils::{FractionalNoiseGenerationMethod, Generator, NoiseGenerationMethod},
};
use ndarray::Array1;
use rayon::prelude::*;

pub struct Fbm {
  #[allow(dead_code)]
  hurst: f64,
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

    match method.unwrap_or(NoiseGenerationMethod::Fft(
      crate::utils::FractionalNoiseGenerationMethod::Kroese,
    )) {
      NoiseGenerationMethod::Fft(FractionalNoiseGenerationMethod::Kroese) => Self {
        hurst,
        n,
        m,
        method: NoiseGenerationMethod::Fft(FractionalNoiseGenerationMethod::Kroese),
        fgn_fft: Some(FgnFft::new(
          hurst,
          n - 1,
          t,
          None,
          Some(FractionalNoiseGenerationMethod::Kroese),
        )),
        fgn_cholesky: None,
      },
      NoiseGenerationMethod::Fft(FractionalNoiseGenerationMethod::DaviesHarte) => Self {
        hurst,
        n,
        m,
        method: NoiseGenerationMethod::Fft(FractionalNoiseGenerationMethod::DaviesHarte),
        fgn_fft: Some(FgnFft::new(
          hurst,
          n - 1,
          t,
          None,
          Some(FractionalNoiseGenerationMethod::DaviesHarte),
        )),
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
      NoiseGenerationMethod::Fft(FractionalNoiseGenerationMethod::Kroese) => {
        self.fgn_fft.as_ref().unwrap().sample()
      }
      NoiseGenerationMethod::Fft(FractionalNoiseGenerationMethod::DaviesHarte) => {
        self.fgn_fft.as_ref().unwrap().sample()
      }
      NoiseGenerationMethod::Cholesky => self.fgn_cholesky.as_ref().unwrap().sample(),
    };

    let mut fbm = Array1::<f64>::zeros(self.n);

    for i in 1..self.n {
      fbm[i] = fbm[i - 1] + fgn[i - 1];
    }

    fbm.to_vec()
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
