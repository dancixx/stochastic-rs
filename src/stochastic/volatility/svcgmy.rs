use impl_new_derive::ImplNew;
use ndarray::{s, Array1};
use ndarray_rand::RandomExt;
use rand::Rng;
use rand_distr::{Exp, Uniform};
use scilib::math::basic::gamma;

use crate::{
  stats::non_central_chi_squared,
  stochastic::{process::poisson::Poisson, Sampling},
};

#[derive(ImplNew)]
pub struct SVCGMY {
  /// Positive jump rate lambda_plus (corresponds to G)
  pub lambda_plus: f64, // G
  /// Negative jump rate lambda_minus (corresponds to M)
  pub lambda_minus: f64, // M
  /// Jump activity parameter alpha (corresponds to Y), with 0 < alpha < 2
  pub alpha: f64,
  ///
  pub kappa: f64,
  ///
  pub eta: f64,
  ///
  pub zeta: f64,
  ///
  pub rho: f64,
  /// Number of time steps
  pub n: usize,
  /// Jumps
  pub j: usize,
  ///
  pub x0: Option<f64>,
  /// Initial value
  pub v0: Option<f64>,
  /// Total time horizon
  pub t: Option<f64>,
  /// Number of samples for parallel sampling (not used in this implementation)
  pub m: Option<usize>,
}

impl Sampling<f64> for SVCGMY {
  fn sample(&self) -> Array1<f64> {
    let t_max = self.t.unwrap_or(1.0);
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let mut x = Array1::<f64>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    let C = (gamma(2.0 - self.alpha)
      * (self.lambda_plus.powf(self.alpha - 2.0) + self.lambda_minus.powf(self.alpha - 2.0)))
    .powi(-1);
    let c = (2.0 * self.kappa) / ((1.0 - (-self.kappa * dt).exp()) * self.zeta.powi(2));

    let mut v = Array1::<f64>::zeros(self.n);
    v[0] = self.v0.unwrap_or(0.0);

    for i in 1..self.n {
      let xi = non_central_chi_squared::sample(
        4.0 * self.kappa * self.eta / self.zeta.powi(2),
        2.0 * c * v[i - 1] * (-self.kappa * dt).exp(),
      );

      v[i] = xi / (2.0 * c);
    }

    let U = Array1::<f64>::random(self.j, Uniform::new(0.0, 1.0));
    let E = Array1::<f64>::random(self.j, Exp::new(1.0).unwrap());
    let E_ = Array1::<f64>::random(self.j, Exp::new(1.0).unwrap());
    let P = Poisson::new(1.0, Some(self.j), None, None);
    let mut P = P.sample();

    for (idx, p) in P.iter_mut().enumerate() {
      *p = *p + E_[idx];
    }

    let tau = Array1::<f64>::random(self.j, Uniform::new(0.0, t_max));
    let mut c_t = Array1::<f64>::zeros(self.j);

    for j in 1..self.j {
      c_t[j] = C * v.slice(s![..j - 1]).sum()
    }

    let mut rng = rand::thread_rng();

    for i in 1..self.n {
      let mut jump_component = 0.0;

      for j in 1..self.j {
        let v_j = if rng.gen_bool(0.5) {
          self.lambda_plus
        } else {
          -self.lambda_minus
        };

        let term1 = ((self.alpha * P[j]) / (2.0 * c_t[j] * t_max)).powf(-1.0 / self.alpha);
        let term2 = E[j] * U[j].powf(1.0 / self.alpha) / v_j.abs();
        let jump_size = term1.min(term2) * (v_j / v_j.abs());

        jump_component += jump_size;
      }

      let b = -(v[i]
        * (self.lambda_plus.powf(self.alpha - 1.0) - self.lambda_minus.powf(self.alpha - 1.0))
        / ((1.0 - self.alpha)
          * (self.lambda_plus.powf(self.alpha - 2.0) + self.lambda_minus.powf(self.alpha) - 2.0)));

      x[i] = x[i - 1] + jump_component + b * dt + self.rho * v[i - 1];
    }

    x
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{plot_1d, stochastic::N};

  #[test]
  fn svcgmy_length_equals_n() {
    let svcgmy = SVCGMY::new(
      25.46,
      4.604,
      0.52,
      1.003,
      0.0711,
      0.3443,
      -2.0280,
      N,
      1024,
      None,
      Some(0.0064),
      Some(1.0),
      None,
    );
    assert_eq!(svcgmy.sample().len(), N);
  }

  #[test]
  fn svcgmy_starts_with_x0() {
    let svcgmy = SVCGMY::new(
      25.46,
      4.604,
      0.52,
      1.003,
      0.0711,
      0.3443,
      -2.0280,
      N,
      1024,
      None,
      Some(0.0064),
      Some(1.0),
      None,
    );
    assert_eq!(svcgmy.sample()[0], 0.0);
  }

  #[test]
  fn svcgmy_plot() {
    let svcgmy = SVCGMY::new(
      25.46,
      4.604,
      0.52,
      1.003,
      0.0711,
      0.3443,
      -2.0280,
      100,
      1024,
      None,
      Some(0.0064),
      Some(1.0),
      None,
    );
    plot_1d!(svcgmy.sample(), "CGMY Process");
  }
}
