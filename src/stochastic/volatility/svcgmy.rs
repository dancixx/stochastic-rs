use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::Rng;
use rand_distr::{Exp, Uniform};
use scilib::math::basic::gamma;

use crate::{
  stats::non_central_chi_squared,
  stochastic::{process::poisson::Poisson, Sampling},
};

/// CGMY Stochastic Volatility process
///
/// https://www.econstor.eu/bitstream/10419/239493/1/175133161X.pdf
#[derive(ImplNew)]
pub struct SVCGMY {
  /// Positive jump rate lambda_plus (corresponds to G)
  pub lambda_plus: f64, // G
  /// Negative jump rate lambda_minus (corresponds to M)
  pub lambda_minus: f64, // M
  /// Jump activity parameter alpha (corresponds to Y), with 0 < alpha < 2
  pub alpha: f64,
  /// Mean reversion rate
  pub kappa: f64,
  /// Long-term volatility
  pub eta: f64,
  /// Volatility of volatility
  pub zeta: f64,
  ///
  pub rho: f64,
  /// Number of time steps
  pub n: usize,
  /// Jumps
  pub j: usize,
  /// Initial value
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
    let mut rng = rand::thread_rng();

    let t_max = self.t.unwrap_or(1.0);
    let dt = t_max / (self.n - 1) as f64;

    let mut x = Array1::<f64>::zeros(self.n);
    let mut v = Array1::<f64>::zeros(self.n);
    let mut y = Array1::<f64>::zeros(self.n);

    x[0] = self.x0.unwrap_or(0.0);
    v[0] = self.v0.unwrap_or(0.0);

    let C = 1.0
      / (gamma(2.0 - self.alpha)
        * (self.lambda_plus.powf(self.alpha - 2.0) + self.lambda_minus.powf(self.alpha - 2.0)));
    let c = (2.0 * self.kappa) / ((1.0 - (-self.kappa * dt).exp()) * self.zeta.powi(2));
    let df = 4.0 * self.kappa * self.eta / self.zeta.powi(2);

    // Volatilitás folyamat generálása
    for i in 1..self.n {
      let ncp = 2.0 * c * v[i - 1] * (-self.kappa * dt).exp();
      let xi = non_central_chi_squared::sample(df, ncp, &mut rng);
      v[i] = xi / (2.0 * c);
    }

    let U = Array1::<f64>::random(self.j, Uniform::new(0.0, 1.0));
    let E = Array1::random(self.j, Exp::new(1.0).unwrap());
    let P = Poisson::new(1.0, Some(self.j), None, None);
    let P = P.sample();
    let tau = Array1::<f64>::random(self.j, Uniform::new(0.0, 1.0)) * t_max;

    let mut c_tau = Array1::<f64>::zeros(self.j);
    for (idx, tau_j) in tau.iter().enumerate() {
      let k = ((tau_j / dt).ceil() as usize).min(self.n - 1);
      let v_k = if k == 0 { v[0] } else { v[k - 1] };
      c_tau[idx] = C * v_k;
    }

    for i in 1..self.n {
      let numerator = v[i - 1]
        * (self.lambda_plus.powf(self.alpha - 1.0) - self.lambda_minus.powf(self.alpha - 1.0));
      let denominator = (1.0 - self.alpha)
        * (self.lambda_plus.powf(self.alpha - 2.0) + self.lambda_minus.powf(self.alpha - 2.0));
      let b = -numerator / denominator;

      let mut jump_component = 0.0;

      let t_1 = (i - 1) as f64 * dt;
      let t = i as f64 * dt;

      for j in 0..self.j {
        if tau[j] > t_1 && tau[j] <= t {
          let v_j = if rng.gen_bool(0.5) {
            self.lambda_plus
          } else {
            -self.lambda_minus
          };

          let term1 = ((self.alpha * P[j]) / (2.0 * c_tau[j] * t_max)).powf(-1.0 / self.alpha);
          let term2 = E[j] * U[j].powf(1.0 / self.alpha) / v_j.abs();
          let min_term = term1.min(term2);
          let jump_size = min_term * (v_j / v_j.abs());
          jump_component += jump_size;
        }
      }

      y[i] = y[i - 1] + jump_component + b * dt;
    }

    for i in 0..self.n {
      x[i] = y[i] + self.rho * v[i];
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
  use ndarray::Axis;

  use super::*;
  use crate::{plot_1d, plot_nd, stochastic::N};

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
      1000,
      1024,
      Some(-0.25),
      Some(0.0064),
      Some(1.0),
      None,
    );
    plot_1d!(svcgmy.sample(), "SVCGMY Process");
  }

  #[test]
  fn svcgmy_plot_multi() {
    let svcgmy = SVCGMY::new(
      25.46,
      4.604,
      0.52,
      1.003,
      0.0711,
      0.3443,
      -2.0280,
      1000,
      1024,
      Some(-0.25),
      Some(0.0064),
      Some(1.0),
      Some(10),
    );
    plot_nd!(svcgmy.sample_par(), "SVCGMY Process");
  }
}
