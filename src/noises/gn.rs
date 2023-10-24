use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

pub fn gn(n: usize, t: Option<f64>) -> Vec<f64> {
  let sqrt_dt = (t.unwrap_or(1.0) / n as f64).sqrt();
  thread_rng()
    .sample_iter::<f64, StandardNormal>(StandardNormal)
    .take(n)
    .map(|z| z * sqrt_dt)
    .collect()
}
