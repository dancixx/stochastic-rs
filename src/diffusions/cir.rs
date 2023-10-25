use crate::{
  noises::{fgn::FgnFft, gn},
  utils::Generator,
};
use ndarray::Array1;

pub fn cir(
  theta: f64,
  mu: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  use_sym: Option<bool>,
) -> Vec<f64> {
  if 2.0 * theta * mu < sigma.powi(2) {
    panic!("2 * theta * mu < sigma^2")
  }

  let gn = gn::gn(n - 1, Some(t.unwrap_or(1.0)));
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut cir = Array1::<f64>::zeros(n + 1);
  cir[0] = x0.unwrap_or(0.0);

  for (i, dw) in gn.iter().enumerate() {
    let random = match use_sym.unwrap_or(false) {
      true => sigma * (cir[i]).abs().sqrt() * dw,
      false => sigma * (cir[i]).max(0.0).sqrt() * dw,
    };
    cir[i + 1] = cir[i] + theta * (mu - cir[i]) * dt + random
  }

  cir.to_vec()
}

#[allow(clippy::too_many_arguments)]
pub fn fcir(
  hurst: f64,
  theta: f64,
  mu: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  use_sym: Option<bool>,
) -> Vec<f64> {
  if !(0.0..1.0).contains(&hurst) {
    panic!("Hurst parameter must be in (0, 1)")
  }

  if 2.0 * theta * mu < sigma.powi(2) {
    panic!("2 * theta * mu < sigma^2")
  }

  let fgn = FgnFft::new(hurst, n - 1, t, None, None).sample();
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut fcir = Array1::<f64>::zeros(n + 1);
  fcir[0] = x0.unwrap_or(0.0);

  for (i, dw) in fgn.iter().enumerate() {
    let random = match use_sym.unwrap_or(false) {
      true => sigma * (fcir[i]).abs().sqrt() * dw,
      false => sigma * (fcir[i]).max(0.0) * dw,
    };
    fcir[i + 1] = fcir[i] + theta * (mu - fcir[i]) * dt + random
  }

  fcir.to_vec()
}
