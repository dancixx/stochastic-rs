use crate::{
  noises::{fgn::FgnFft, gn},
  utils::Generator,
};
use ndarray::Array2;

// TODO: under development
pub fn correlated_bms(rho: f64, n: usize, t: Option<f64>) -> Vec<Vec<f64>> {
  if !(-1.0..=1.0).contains(&rho) {
    panic!("Correlation coefficient must be in [-1, 1]");
  }

  let mut bms = Array2::<f64>::zeros((n, 2));

  let gn1 = gn::gn(n - 1, Some(t.unwrap_or(1.0)));
  let gn2 = gn::gn(n - 1, Some(t.unwrap_or(1.0)));

  for i in 1..n {
    bms[[i, 0]] = bms[[i - 1, 0]] + gn1[i - 1];
    bms[[i, 1]] = rho * gn1[i - 1] + (1.0 - rho.powi(2)).sqrt() * gn2[i - 1];
  }

  bms.outer_iter().map(|x| x.to_vec()).collect()
}

// TODO: under development
pub fn correlated_fbms(
  hurst1: f64,
  hurst2: f64,
  rho: f64,
  n: usize,
  t: Option<f64>,
) -> Vec<Vec<f64>> {
  if !(-1.0..=1.0).contains(&rho) || !(0.0..1.0).contains(&hurst1) || !(0.0..1.0).contains(&hurst2)
  {
    panic!("Correlation coefficient must be in [-1, 1] and Hurst parameters must be in (0, 1)");
  }

  let mut fbms = Array2::<f64>::zeros((n, 2));

  let fgn1 = FgnFft::new(hurst1, n - 1, t, None).sample();
  let fgn2 = FgnFft::new(hurst2, n - 1, t, None).sample();

  for i in 1..n {
    fbms[[i, 0]] = fbms[[i - 1, 0]] + fgn1[i - 1];
    fbms[[i, 1]] = rho * fgn2[i - 1] + (1.0 - rho.powi(2)).sqrt() * fgn2[i - 1];
  }

  fbms.outer_iter().map(|x| x.to_vec()).collect()
}
