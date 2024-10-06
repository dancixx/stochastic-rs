use impl_new_derive::ImplNew;
use ndarray::{s, Array1};

use crate::stochastic::{
  noise::fgn::FGN, process::cpoisson::CompoundPoisson, ProcessDistribution, Sampling, Sampling3D,
};

#[derive(ImplNew)]
pub struct JumpFOU<D>
where
  D: ProcessDistribution,
{
  pub mu: f64,
  pub sigma: f64,
  pub theta: f64,
  pub lambda: Option<f64>,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub jump_distribution: D,
  pub fgn: FGN,
  pub cpoisson: CompoundPoisson<D>,
}

impl<D: ProcessDistribution> Sampling<f64> for JumpFOU<D> {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let fgn = self.fgn.sample();
    let mut jump_fou = Array1::<f64>::zeros(self.n);
    jump_fou[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();

      jump_fou[i] = jump_fou[i - 1]
        + self.theta * (self.mu - jump_fou[i - 1]) * dt
        + self.sigma * fgn[i - 1]
        + jumps.sum();
    }

    jump_fou.slice(s![..self.n()]).to_owned()
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}
