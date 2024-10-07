use impl_new_derive::ImplNew;
use ndarray::{s, Array1};
use rand_distr::Distribution;

use crate::stochastic::{
  noise::fgn::FGN, process::cpoisson::CompoundPoisson, Sampling, Sampling3D,
};

#[derive(ImplNew)]
pub struct JumpFOU<D>
where
  D: Distribution<f64> + Send + Sync,
{
  pub mu: f64,
  pub sigma: f64,
  pub theta: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub fgn: FGN,
  pub cpoisson: CompoundPoisson<D>,
}

impl<D> Sampling<f64> for JumpFOU<D>
where
  D: Distribution<f64> + Send + Sync,
{
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

#[cfg(test)]
mod tests {
  use rand_distr::Normal;

  use crate::{
    plot_1d,
    stochastic::{process::poisson::Poisson, N, X0},
  };

  use super::*;

  #[test]
  fn jump_fou_length_equals_n() {
    let jump_fou = JumpFOU::new(
      2.25,
      2.5,
      1.0,
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::new(0.7, N, None, None),
      CompoundPoisson::new(
        None,
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64), None),
      ),
    );

    assert_eq!(jump_fou.sample().len(), N);
  }

  #[test]
  fn jump_fou_starts_with_x0() {
    let jump_fou = JumpFOU::new(
      2.25,
      2.5,
      1.0,
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::new(0.7, N, None, None),
      CompoundPoisson::new(
        None,
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64), None),
      ),
    );

    assert_eq!(jump_fou.sample()[0], X0);
  }

  #[test]
  fn jump_fou_plot() {
    let jump_fou = JumpFOU::new(
      2.25,
      2.5,
      1.0,
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::new(0.7, N, None, None),
      CompoundPoisson::new(
        None,
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64), None),
      ),
    );

    plot_1d!(jump_fou.sample(), "Jump FOU process");
  }

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn jump_fou_malliavin() {
    unimplemented!()
  }
}
