use ndarray::Array1;

use crate::{
  noises::{fgn_cholesky, fgn_fft, gn},
  utils::NoiseGenerationMethod,
};

pub fn jacobi(
  alpha: f64,
  beta: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Vec<f64> {
  if alpha < 0.0 || beta < 0.0 || sigma < 0.0 {
    panic!("alpha, beta, and sigma must be positive")
  }

  if alpha > beta {
    panic!("alpha must be less than beta")
  }

  let gn = gn::gn(n - 1, Some(t.unwrap_or(1.0)));
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut jacobi = Array1::<f64>::zeros(n + 1);
  jacobi[0] = x0.unwrap_or(0.0);

  for (i, dw) in gn.iter().enumerate() {
    jacobi[i + 1] = match jacobi[i] {
      _ if jacobi[i] <= 0.0 && i > 0 => 0.0,
      _ if jacobi[i] >= 1.0 && i > 0 => 1.0,
      _ => {
        jacobi[i]
          + (alpha - beta * jacobi[i]) * dt
          + sigma * (jacobi[i] * (1.0 - jacobi[i])).sqrt() * dw
      }
    }
  }

  jacobi.to_vec()
}

#[allow(clippy::too_many_arguments)]
pub fn fjacobi(
  hurst: f64,
  alpha: f64,
  beta: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  method: Option<NoiseGenerationMethod>,
) -> Vec<f64> {
  if !(0.0..1.0).contains(&hurst) {
    panic!("Hurst parameter must be in (0, 1)")
  }

  if alpha < 0.0 || beta < 0.0 || sigma < 0.0 {
    panic!("alpha, beta, and sigma must be positive")
  }

  if alpha > beta {
    panic!("alpha must be less than beta")
  }

  let fgn = match method.unwrap_or(NoiseGenerationMethod::Fft) {
    NoiseGenerationMethod::Fft => fgn_fft::fgn(hurst, n, t),
    NoiseGenerationMethod::Cholesky => fgn_cholesky::fgn(hurst, n - 1, t),
  };
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut fjacobi = Array1::<f64>::zeros(n + 1);
  fjacobi[0] = x0.unwrap_or(0.0);

  for (i, dw) in fgn.iter().enumerate() {
    fjacobi[i + 1] = match fjacobi[i] {
      _ if fjacobi[i] <= 0.0 && i > 0 => 0.0,
      _ if fjacobi[i] >= 1.0 && i > 0 => 1.0,
      _ => {
        fjacobi[i]
          + (alpha - beta * fjacobi[i]) * dt
          + sigma * (fjacobi[i] * (1.0 - fjacobi[i])).sqrt() * dw
      }
    }
  }

  fjacobi.to_vec()
}
