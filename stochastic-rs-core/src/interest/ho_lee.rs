use std::sync::Arc;

use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::Sampling;

#[allow(non_snake_case)]
#[derive(Default)]
pub struct HoLee<'a>
where
  'a: 'static,
{
  pub f_T: Option<Arc<dyn Fn(f64) -> f64 + Send + Sync + 'a>>,
  pub theta: Option<f64>,
  pub sigma: f64,
  pub n: usize,
  pub t: f64,
  pub m: Option<usize>,
}

impl<'a> HoLee<'a> {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    Self {
      f_T: params.f_T.clone(),
      theta: params.theta,
      sigma: params.sigma,
      n: params.n,
      t: params.t,
      m: params.m,
    }
  }
}

impl<'a> Sampling<f64> for HoLee<'a> {
  fn sample(&self) -> Array1<f64> {
    assert!(
      self.theta.is_none() && self.f_T.is_none(),
      "theta or f_T must be provided"
    );
    let dt = self.t / self.n as f64;
    let gn = Array1::random(
      self.n,
      Normal::new(0.0, (self.t / self.n as f64).sqrt()).unwrap(),
    );

    let mut r = Array1::<f64>::zeros(self.n + 1);

    for i in 1..=self.n {
      let drift = if let Some(r#fn) = self.f_T.as_ref() {
        (r#fn)(i as f64 * dt) + self.sigma.powf(2.0)
      } else {
        self.theta.unwrap() + self.sigma.powf(2.0)
      };

      r[i] = r[i - 1] + drift * dt + self.sigma * gn[i - 1];
    }

    r
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
