use crate::{
  noises::{fgn::FgnFft, gn::gn},
  utils::Generator,
};
use ndarray::Array1;

pub fn gbm(mu: f32, sigma: f32, n: usize, x0: Option<f32>, t: Option<f32>) -> Vec<f32> {
  let gn = gn(n - 1, Some(t.unwrap_or(1.0)));
  let dt = t.unwrap_or(1.0) / n as f32;

  let mut gbm = Array1::<f32>::zeros(n);
  gbm[0] = x0.unwrap_or(100.0);

  for i in 1..n {
    gbm[i] = gbm[i - 1] + mu * gbm[i - 1] * dt + sigma * gbm[i - 1] * gn[i - 1]
  }

  gbm.to_vec()
}

pub fn fgbm(
  hurst: f32,
  mu: f32,
  sigma: f32,
  n: usize,
  x0: Option<f32>,
  t: Option<f32>,
) -> Vec<f32> {
  if !(0.0..1.0).contains(&hurst) {
    panic!("Hurst parameter must be in (0, 1)")
  }

  let fgn = FgnFft::new(hurst, n - 1, t, None).sample();
  let dt = t.unwrap_or(1.0) / n as f32;

  let mut fgbm = Array1::<f32>::zeros(n);
  fgbm[0] = x0.unwrap_or(100.0);

  for i in 1..n {
    fgbm[i] = fgbm[i - 1] + mu * fgbm[i - 1] * dt + sigma * fgbm[i - 1] * fgn[i - 1]
  }

  fgbm.to_vec()
}
