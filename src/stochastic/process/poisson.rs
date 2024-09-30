use ndarray::{Array0, Array1, Axis, Dim};
use ndarray_rand::rand_distr::{Distribution, Exp};
use ndarray_rand::RandomExt;
use rand::thread_rng;

use crate::stochastic::Sampling;

#[derive(Default)]
pub struct Poisson {
  pub lambda: f64,
  pub n: Option<usize>,
  pub t_max: Option<f64>,
  pub m: Option<usize>,
}

impl Poisson {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    Self {
      lambda: params.lambda,
      n: params.n,
      t_max: params.t_max,
      m: params.m,
    }
  }
}

impl Sampling<f64> for Poisson {
  fn sample(&self) -> Array1<f64> {
    if let Some(n) = self.n {
      let exponentials = Array1::random(n, Exp::new(1.0 / self.lambda).unwrap());
      let mut poisson = Array1::<f64>::zeros(n + 1);
      for i in 1..(n + 1) {
        poisson[i] = poisson[i - 1] + exponentials[i - 1];
      }

      poisson
    } else if let Some(t_max) = self.t_max {
      let mut poisson = Array1::from(vec![0.0]);
      let mut t = 0.0;

      while t < t_max {
        t += Exp::new(1.0 / self.lambda)
          .unwrap()
          .sample(&mut thread_rng());
        poisson
          .push(Axis(0), Array0::from_elem(Dim(()), t).view())
          .unwrap();
      }

      poisson
    } else {
      panic!("n or t_max must be provided");
    }
  }

  fn n(&self) -> usize {
    self.n.unwrap_or(0)
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
