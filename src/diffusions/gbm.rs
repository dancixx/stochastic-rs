use crate::{
  noises::{fgn::FgnFft, gn::gn},
  utils::Generator,
};
use ndarray::Array1;

pub fn gbm(mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>) -> Vec<f64> {
  let gn = gn(n - 1, Some(t.unwrap_or(1.0)));
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut gbm = Array1::<f64>::zeros(n);
  gbm[0] = x0.unwrap_or(100.0);

  for i in 1..n {
    gbm[i] = gbm[i - 1] + mu * gbm[i - 1] * dt + sigma * gbm[i - 1] * gn[i - 1]
  }

  gbm.to_vec()
}

pub fn fgbm(
  hurst: f64,
  mu: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Vec<f64> {
  if !(0.0..1.0).contains(&hurst) {
    panic!("Hurst parameter must be in (0, 1)")
  }

  let fgn = FgnFft::new(hurst, n - 1, t, None).sample();
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut fgbm = Array1::<f64>::zeros(n);
  fgbm[0] = x0.unwrap_or(100.0);

  for i in 1..n {
    fgbm[i] = fgbm[i - 1] + mu * fgbm[i - 1] * dt + sigma * fgbm[i - 1] * fgn[i - 1]
  }

  fgbm.to_vec()
}
