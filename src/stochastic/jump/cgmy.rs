use crate::stochastic::{process::poisson::Poisson, Sampling};
use impl_new_derive::ImplNew;
use ndarray::Array1;
use rand::{thread_rng, Rng};
use rand_distr::Exp;
use scilib::math::basic::gamma;

/// CGMY process
///
/// The CGMY process is a pure jump Lévy process used in financial modeling to capture the dynamics
/// of asset returns with jumps and heavy tails. It is characterized by four parameters:
/// `C`, `G` (lambda_plus), `M` (lambda_minus), and `Y` (alpha).
///
/// The process is defined by the Lévy measure:
///
/// \[ \nu(x) = C \frac{e^{-G x}}{x^{1 + Y}} 1_{x > 0} + C \frac{e^{M x}}{|x|^{1 + Y}} 1_{x < 0} \]
///
/// where:
/// - `c` (C) > 0 controls the overall intensity of the jumps.
/// - `lambda_plus` (G) > 0 is the rate of exponential decay of positive jumps.
/// - `lambda_minus` (M) > 0 is the rate of exponential decay of negative jumps.
/// - `alfa` (Y), with 0 < `alfa` < 2, controls the jump activity (number of small jumps).
///
/// Series representation of the CGMY process:
/// \[ X(t) = \sum_{i=1}^{\infty} ((alpha * Gamma_j / 2C)^(-1/alpha) \land E_j * U_j^(1/alpha) * abs(V_j)^-1)) * V_j / |V_j| 1_[0, t] + b_t * t \]
///
/// This implementation simulates the CGMY process using a discrete approximation over a grid of time points.
/// At each time step, we generate a Poisson random number of jumps, and for each jump, we generate the jump size
/// according to the CGMY process. The process also includes a drift component computed from the parameters.
///
/// # References
///
/// - Cont, R., & Tankov, P. (2004). *Financial Modelling with Jump Processes*. Chapman and Hall/CRC.
/// - Madan, D. B., Carr, P., & Chang, E. C. (1998). The Variance Gamma Process and Option Pricing. *European Finance Review*, 2(1), 79-105.
/// https://www.econstor.eu/bitstream/10419/239493/1/175133161X.pdf
///
#[derive(ImplNew)]
pub struct CGMY {
  /// Positive jump rate lambda_plus (corresponds to G)
  pub lambda_plus: f64, // G
  /// Negative jump rate lambda_minus (corresponds to M)
  pub lambda_minus: f64, // M
  /// Jump activity parameter alpha (corresponds to Y), with 0 < alpha < 2
  pub alpha: f64,
  /// Number of time steps
  pub n: usize,
  /// Jumps
  pub j: usize,
  /// Initial value
  pub x0: Option<f64>,
  /// Total time horizon
  pub t: Option<f64>,
  /// Number of samples for parallel sampling (not used in this implementation)
  pub m: Option<usize>,
}

impl Sampling<f64> for CGMY {
  fn sample(&self) -> Array1<f64> {
    let t_max = self.t.unwrap_or(1.0);
    let dt = t_max / (self.n - 1) as f64;
    let mut x = Array1::<f64>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    let c = (gamma(2.0 - self.alpha)
      * (self.lambda_plus.powf(self.alpha - 2.0) + self.lambda_minus.powf(self.alpha - 2.0)))
    .powi(-1);

    let b_t = -c
      * gamma(1.0 - self.alpha)
      * (self.lambda_plus.powf(self.alpha - 1.0) - self.lambda_minus.powf(self.alpha - 1.0));

    let mut rng = thread_rng();
    let poisson = Poisson::new(1.0, Some(self.j), None, None);

    for i in 1..self.n {
      let mut jump_component = 0.0;

      let poisson = poisson.sample();
      for j in 0..self.j {
        let u_j: f64 = rng.gen();
        let e_j: f64 = rng.sample(Exp::new(1.0).unwrap());

        let v_j = if rng.gen_bool(0.5) {
          self.lambda_plus
        } else {
          -self.lambda_minus
        };

        let term1 = (self.alpha * poisson[j] / (2.0 * c)).powf(-1.0 / self.alpha);
        let term2 = e_j * u_j.powf(1.0 / self.alpha) / v_j.abs();

        let jump_size = term1.min(term2) * (v_j / v_j.abs());

        jump_component += jump_size;
      }

      x[i] = x[i - 1] + b_t * dt + jump_component;
    }

    x
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
  use super::*;
  use crate::{plot_1d, stochastic::N};

  #[test]
  fn cgmy_length_equals_n() {
    let cgmy = CGMY {
      lambda_plus: 5.0,
      lambda_minus: 5.0,
      alpha: 0.7,
      n: N,
      j: 1000,
      x0: Some(0.0),
      t: Some(1.0),
      m: None,
    };

    assert_eq!(cgmy.sample().len(), N);
  }

  #[test]
  fn cgmy_starts_with_x0() {
    let cgmy = CGMY {
      lambda_plus: 5.0,
      lambda_minus: 5.0,
      alpha: 0.7,
      n: N,
      j: 1000,
      x0: Some(0.0),
      t: Some(1.0),
      m: None,
    };

    assert_eq!(cgmy.sample()[0], 0.0);
  }

  #[test]
  fn cgmy_plot() {
    let cgmy = CGMY {
      lambda_plus: 5.0,
      lambda_minus: 5.0,
      alpha: 0.7,
      n: N,
      j: 1000,
      x0: Some(0.0),
      t: Some(1.0),
      m: None,
    };

    // Plot the CGMY sample path
    plot_1d!(cgmy.sample(), "CGMY Process");
  }
}
