use crate::{
  noises::{fgn::FgnFft, gn},
  utils::Generator,
};
use ndarray::Array1;

pub fn ou(mu: f64, sigma: f64, theta: f64, n: usize, x0: Option<f64>, t: Option<f64>) -> Vec<f64> {
  let gn = gn::gn(n - 1, Some(t.unwrap_or(1.0)));
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut ou = Array1::<f64>::zeros(n);
  ou[0] = x0.unwrap_or(0.0);

  for i in 1..n {
    ou[i] = ou[i - 1] + theta * (mu - ou[i - 1]) * dt + sigma * gn[i - 1]
  }

  ou.to_vec()
}

#[allow(clippy::too_many_arguments)]
pub fn fou(
  hurst: f64,
  mu: f64,
  sigma: f64,
  theta: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Vec<f64> {
  if !(0.0..1.0).contains(&hurst) {
    panic!("Hurst parameter must be in (0, 1)")
  }

  let fgn = FgnFft::new(hurst, n - 1, t, None).sample();
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut fou = Array1::<f64>::zeros(n);
  fou[0] = x0.unwrap_or(0.0);

  for i in 1..n {
    fou[i] = fou[i - 1] + theta * (mu - fou[i - 1]) * dt + sigma * fgn[i - 1]
  }

  fou.to_vec()
}
