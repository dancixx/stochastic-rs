use crate::prelude::*;
use rayon::prelude::*;

pub fn par_gn(m: usize, n: usize, t: Option<f64>) -> Vec<Vec<f64>> {
  (0..m)
    .into_par_iter()
    .map(|_| gn::gn(n, t))
    .collect::<Vec<Vec<f64>>>()
}

pub fn par_fgn_fft(m: usize, hurst: f64, n: usize, t: Option<f64>) -> Vec<Vec<f64>> {
  (0..m)
    .into_par_iter()
    .map(|_| fgn_fft::fgn(hurst, n, t))
    .collect::<Vec<Vec<f64>>>()
}

pub fn par_fgn_cholesky(m: usize, hurst: f64, n: usize, t: Option<f64>) -> Vec<Vec<f64>> {
  (0..m)
    .into_par_iter()
    .map(|_| fgn_cholesky::fgn(hurst, n, t))
    .collect::<Vec<Vec<f64>>>()
}

pub fn par_bm(m: usize, n: usize, t: Option<f64>) -> Vec<Vec<f64>> {
  (0..m)
    .into_par_iter()
    .map(|_| bm::bm(n, t))
    .collect::<Vec<Vec<f64>>>()
}

pub fn par_fbm(
  m: usize,
  hurst: f64,
  n: usize,
  t: Option<f64>,
  method: Option<NoiseGenerationMethod>,
) -> Vec<Vec<f64>> {
  (0..m)
    .into_par_iter()
    .map(|_| fbm::fbm(hurst, n, t, method))
    .collect::<Vec<Vec<f64>>>()
}

pub fn par_cir(
  m: usize,
  theta: f64,
  mu: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  use_sym: Option<bool>,
) -> Vec<Vec<f64>> {
  (0..m)
    .into_par_iter()
    .map(|_| cir::cir(theta, mu, sigma, n, x0, t, use_sym))
    .collect::<Vec<Vec<f64>>>()
}

pub fn par_fcir(
  m: usize,
  hurst: f64,
  theta: f64,
  mu: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  method: Option<NoiseGenerationMethod>,
  use_sym: Option<bool>,
) -> Vec<Vec<f64>> {
  (0..m)
    .into_par_iter()
    .map(|_| cir::fcir(hurst, theta, mu, sigma, n, x0, t, method, use_sym))
    .collect::<Vec<Vec<f64>>>()
}

pub fn par_ou(
  m: usize,
  theta: f64,
  mu: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Vec<Vec<f64>> {
  (0..m)
    .into_par_iter()
    .map(|_| ou::ou(mu, sigma, theta, n, x0, t))
    .collect::<Vec<Vec<f64>>>()
}

pub fn par_fou(
  m: usize,
  hurst: f64,
  theta: f64,
  mu: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  method: Option<NoiseGenerationMethod>,
) -> Vec<Vec<f64>> {
  (0..m)
    .into_par_iter()
    .map(|_| ou::fou(hurst, mu, sigma, theta, n, x0, t, method))
    .collect::<Vec<Vec<f64>>>()
}

pub fn par_gbm(
  m: usize,
  mu: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Vec<Vec<f64>> {
  (0..m)
    .into_par_iter()
    .map(|_| gbm::gbm(mu, sigma, n, x0, t))
    .collect::<Vec<Vec<f64>>>()
}

pub fn par_fgbm(
  m: usize,
  hurst: f64,
  mu: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  method: Option<NoiseGenerationMethod>,
) -> Vec<Vec<f64>> {
  (0..m)
    .into_par_iter()
    .map(|_| gbm::fgbm(hurst, mu, sigma, n, x0, t, method))
    .collect::<Vec<Vec<f64>>>()
}

pub fn par_jacobi(
  m: usize,
  alpha: f64,
  beta: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Vec<Vec<f64>> {
  (0..m)
    .into_par_iter()
    .map(|_| jacobi::jacobi(alpha, beta, sigma, n, x0, t))
    .collect::<Vec<Vec<f64>>>()
}

pub fn par_fjacobi(
  m: usize,
  hurst: f64,
  alpha: f64,
  beta: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  method: Option<NoiseGenerationMethod>,
) -> Vec<Vec<f64>> {
  (0..m)
    .into_par_iter()
    .map(|_| jacobi::fjacobi(hurst, alpha, beta, sigma, n, x0, t, method))
    .collect::<Vec<Vec<f64>>>()
}
