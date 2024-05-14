use crate::{
  noises::{fgn::FgnFft, gn},
  utils::Generator,
};
use ndarray::Array1;

pub fn cir(
  theta: f32,
  mu: f32,
  sigma: f32,
  n: usize,
  x0: Option<f32>,
  t: Option<f32>,
  use_sym: Option<bool>,
) -> Vec<f32> {
  if 2.0 * theta * mu < sigma.powi(2) {
    panic!("2 * theta * mu < sigma^2")
  }

  let gn = gn::gn(n - 1, Some(t.unwrap_or(1.0)));
  let dt = t.unwrap_or(1.0) / n as f32;

  let mut cir = Array1::<f32>::zeros(n);
  cir[0] = x0.unwrap_or(0.0);

  for i in 1..n {
    let random = match use_sym.unwrap_or(false) {
      true => sigma * (cir[i - 1]).abs().sqrt() * gn[i - 1],
      false => sigma * (cir[i - 1]).max(0.0).sqrt() * gn[i - 1],
    };
    cir[i] = cir[i - 1] + theta * (mu - cir[i - 1]) * dt + random
  }

  cir.to_vec()
}

#[allow(clippy::too_many_arguments)]
pub fn fcir(
  hurst: f32,
  theta: f32,
  mu: f32,
  sigma: f32,
  n: usize,
  x0: Option<f32>,
  t: Option<f32>,
  use_sym: Option<bool>,
) -> Vec<f32> {
  if !(0.0..1.0).contains(&hurst) {
    panic!("Hurst parameter must be in (0, 1)")
  }

  if 2.0 * theta * mu < sigma.powi(2) {
    panic!("2 * theta * mu < sigma^2")
  }

  let fgn = FgnFft::new(hurst, n - 1, t, None).sample();
  let dt = t.unwrap_or(1.0) / n as f32;

  let mut fcir = Array1::<f32>::zeros(n);
  fcir[0] = x0.unwrap_or(0.0);

  for i in 1..n {
    let random = match use_sym.unwrap_or(false) {
      true => sigma * (fcir[i - 1]).abs().sqrt() * fgn[i - 1],
      false => sigma * (fcir[i - 1]).max(0.0) * fgn[i - 1],
    };
    fcir[i] = fcir[i - 1] + theta * (mu - fcir[i - 1]) * dt + random
  }

  fcir.to_vec()
}
