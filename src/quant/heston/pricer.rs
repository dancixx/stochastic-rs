use std::f64::consts::FRAC_1_PI;

use impl_new_derive::ImplNew;
use nalgebra::DVector;
use num_complex::Complex64;
use quadrature::double_exponential;

use crate::quant::r#trait::Pricer;

#[derive(ImplNew, Clone, Default)]
pub struct HestonPricer {
  /// Stock price
  pub s: f64,
  /// Initial volatility
  pub v0: f64,
  /// Strike price
  pub k: f64,
  /// Risk-free rate
  pub r: f64,
  /// Dividend yield
  pub q: Option<f64>,
  /// Correlation between the stock price and its volatility
  pub rho: f64,
  /// Mean reversion rate
  pub kappa: f64,
  /// Long-run average volatility
  pub theta: f64,
  /// Volatility of volatility
  pub sigma: f64,
  /// Market price of volatility risk
  pub lambda: Option<f64>,
  /// Time to maturity
  pub tau: f64,
  /// Evaluation date
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date
  pub expiry: Option<chrono::NaiveDate>,
}

impl Pricer for HestonPricer {
  /// Calculate the price of a European call option using the Heston model
  /// https://quant.stackexchange.com/a/18686
  fn calculate_price(&self) -> (f64, f64) {
    let call = self.s * (-self.q.unwrap_or(0.0) * self.tau).exp() * self.p(1, self.tau)
      - self.k * (-self.r * self.tau).exp() * self.p(2, self.tau);
    let put = call + self.k * (-self.r * self.tau).exp()
      - self.s * (-self.q.unwrap_or(0.0) * self.tau).exp();

    (call, put)
  }

  /// Update the parameters from the calibration
  fn update_params(&mut self, params: DVector<f64>) {
    self.v0 = params[0];
    self.theta = params[1];
    self.rho = params[2];
    self.kappa = params[3];
    self.sigma = params[4];
  }

  /// Update the strike price
  fn update_strike(&mut self, k: f64) {
    self.k = k;
  }

  /// Derivatives
  fn derivatives(&self) -> Vec<f64> {
    vec![
      self.dC_dv0(self.tau),
      self.dC_dtheta(self.tau),
      self.dC_drho(self.tau),
      self.dC_dkappa(self.tau),
      self.dC_dsigma(self.tau),
    ]
  }
}

impl HestonPricer {
  pub(self) fn u(&self, j: u8) -> f64 {
    match j {
      1 => 0.5,
      2 => -0.5,
      _ => panic!("Invalid j"),
    }
  }

  pub(self) fn b(&self, j: u8) -> f64 {
    match j {
      1 => self.kappa + self.lambda.unwrap_or(1.0) - self.rho * self.sigma,
      2 => self.kappa + self.lambda.unwrap_or(1.0),
      _ => panic!("Invalid j"),
    }
  }

  pub(self) fn d(&self, j: u8, phi: f64) -> Complex64 {
    ((self.b(j) - self.rho * self.sigma * phi * Complex64::i()).powi(2)
      - self.sigma.powi(2) * (2.0 * Complex64::i() * self.u(j) * phi - phi.powi(2)))
    .sqrt()
  }

  pub(self) fn g(&self, j: u8, phi: f64) -> Complex64 {
    (self.b(j) - self.rho * self.sigma * Complex64::i() * phi + self.d(j, phi))
      / (self.b(j) - self.rho * self.sigma * Complex64::i() * phi - self.d(j, phi))
  }

  pub(self) fn C(&self, j: u8, phi: f64, tau: f64) -> Complex64 {
    (self.r - self.q.unwrap_or(0.0)) * Complex64::i() * phi * tau
      + (self.kappa * self.theta / self.sigma.powi(2))
        * ((self.b(j) - self.rho * self.sigma * Complex64::i() * phi + self.d(j, phi)) * tau
          - 2.0
            * ((1.0 - self.g(j, phi) * (self.d(j, phi) * tau).exp()) / (1.0 - self.g(j, phi))).ln())
  }

  pub(self) fn D(&self, j: u8, phi: f64, tau: f64) -> Complex64 {
    ((self.b(j) - self.rho * self.sigma * Complex64::i() * phi + self.d(j, phi))
      / self.sigma.powi(2))
      * ((1.0 - (self.d(j, phi) * tau).exp())
        / (1.0 - self.g(j, phi) * (self.d(j, phi) * tau).exp()))
  }

  pub(self) fn f(&self, j: u8, phi: f64, tau: f64) -> Complex64 {
    (self.C(j, phi, tau) + self.D(j, phi, tau) * self.v0 + Complex64::i() * phi * self.s.ln()).exp()
  }

  pub(self) fn re(&self, j: u8, tau: f64) -> impl Fn(f64) -> f64 {
    let self_ = self.clone();
    move |phi: f64| -> f64 {
      (self_.f(j, phi, tau) * (-Complex64::i() * phi * self_.k.ln()).exp() / (Complex64::i() * phi))
        .re
    }
  }

  pub(self) fn p(&self, j: u8, tau: f64) -> f64 {
    0.5 + FRAC_1_PI * double_exponential::integrate(self.re(j, tau), 0.00001, 50.0, 10e-6).integral
  }

  /// Partial derivative of the C function with respect to parameters
  /// https://www.sciencedirect.com/science/article/abs/pii/S0377221717304460

  /// Partial derivative of the C function with respect to the v0 parameter
  pub(crate) fn dC_dv0(&self, tau: f64) -> f64 {
    (-self.A(tau) / self.v0).re
  }

  /// Partial derivative of the C function with respect to the theta parameter
  pub(crate) fn dC_dtheta(&self, tau: f64) -> f64 {
    ((2.0 * self.kappa / self.sigma.powi(2)) * self.D_(tau)
      - self.kappa * self.rho * tau * Complex64::i() * self.u(1) / self.sigma)
      .re
  }

  /// Partial derivative of the C function with respect to the rho parameter
  pub(crate) fn dC_drho(&self, tau: f64) -> f64 {
    (-self.kappa * self.theta * tau * Complex64::i() * self.u(1) / self.sigma).re
  }

  /// Partial derivative of the C function with respect to the kappa parameter
  pub(crate) fn dC_dkappa(&self, tau: f64) -> f64 {
    (2.0 * self.theta * self.D_(tau) / self.sigma.powi(2)
      + ((2.0 * self.kappa * self.theta) / self.sigma.powi(2) * self.B(tau)) * self.dB_dkappa(tau)
      - (self.theta * self.rho * tau * Complex64::i() * self.u(1) / self.sigma))
      .re
  }

  /// Partial derivative of the C function with respect to the sigma parameter
  pub(crate) fn dC_dsigma(&self, tau: f64) -> f64 {
    ((-4.0 * self.kappa * self.theta / self.sigma.powi(3)) * self.D_(tau)
      + ((2.0 * self.kappa * self.theta) / (self.sigma.powi(2) * self.d_())) * self.dd_dsigma()
      + self.kappa * self.theta * self.rho * tau * Complex64::i() * self.u(1) / self.sigma.powi(2))
    .re
  }

  pub(self) fn xi(&self) -> Complex64 {
    self.kappa - self.sigma * self.rho * Complex64::i() * self.u(1)
  }

  pub(self) fn d_(&self) -> Complex64 {
    (self.xi().powi(2) + self.sigma.powi(2) * (self.u(1).powi(2) + Complex64::i() * self.u(1)))
      .sqrt()
  }

  pub(self) fn dd_dsigma(&self) -> Complex64 {
    (self.sigma * (self.u(1) + Complex64::i() * self.u(1))) / self.d_()
  }

  pub(self) fn A1(&self, tau: f64) -> Complex64 {
    (self.u(1).powi(2) + Complex64::i() * self.u(1)) * (self.d_() * tau / 2.0).sinh()
  }

  pub(self) fn A2(&self, tau: f64) -> Complex64 {
    (self.d_() / self.v0) * (self.d_() * tau / 2.0).cosh()
      + (self.xi() / self.v0) * (self.d_() * tau / 2.0).sinh()
  }

  pub(self) fn A(&self, tau: f64) -> Complex64 {
    self.A1(tau) / self.A2(tau)
  }

  pub(self) fn D_(&self, tau: f64) -> Complex64 {
    (self.d_() / self.v0).ln() + (self.kappa - self.d_() / 2.0) * tau
      - (((self.d_() + self.xi()) / (2.0 * self.v0))
        + ((self.d_() - self.xi()) / (2.0 * self.v0)) * (-self.d_() * tau).exp())
      .ln()
  }

  pub(self) fn B(&self, tau: f64) -> Complex64 {
    (self.d_() * (self.kappa * tau / 2.0).exp()) / (self.v0 * self.A2(tau))
  }

  pub(self) fn dB_dkappa(&self, tau: f64) -> Complex64 {
    (self.d_() * tau * (self.kappa * tau / 2.0).exp()) / (2.0 * self.v0 * self.A2(tau))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_heston_single_price() {
    let heston = HestonPricer {
      s: 100.0,
      v0: 0.05,
      k: 100.0,
      r: 0.03,
      q: Some(0.02),
      rho: -0.8,
      kappa: 5.0,
      theta: 0.05,
      sigma: 0.5,
      lambda: Some(0.0),
      tau: 0.5,
      ..Default::default()
    };

    let (call, put) = heston.calculate_price();
    println!("Call Price: {}, Put Price: {}", call, put);
  }
}
