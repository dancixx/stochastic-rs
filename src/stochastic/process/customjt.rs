use impl_new_derive::ImplNew;
use ndarray::{Array0, Array1, Axis, Dim};
use ndarray_rand::RandomExt;
use rand::thread_rng;
use rand_distr::Distribution;

use crate::stochastic::Sampling;

#[derive(ImplNew)]
pub struct CustomJt<D>
where
  D: Distribution<f64> + Send + Sync,
{
  pub n: Option<usize>,
  pub t_max: Option<f64>,
  pub m: Option<usize>,
  pub distribution: D,
}

impl<D> Sampling<f64> for CustomJt<D>
where
  D: Distribution<f64> + Send + Sync,
{
  fn sample(&self) -> Array1<f64> {
    if let Some(n) = self.n {
      let random = Array1::random(n, &self.distribution);
      let mut x = Array1::<f64>::zeros(n);
      for i in 1..n {
        x[i] = x[i - 1] + random[i - 1];
      }

      x
    } else if let Some(t_max) = self.t_max {
      let mut x = Array1::from(vec![0.0]);
      let mut t = 0.0;

      while t < t_max {
        t += self.distribution.sample(&mut thread_rng());
        x.push(Axis(0), Array0::from_elem(Dim(()), t).view())
          .unwrap();
      }

      x
    } else {
      panic!("n or t_max must be provided");
    }
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n.unwrap_or(0)
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}
