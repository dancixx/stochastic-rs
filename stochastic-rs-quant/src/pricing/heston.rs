use std::{f64::consts::FRAC_1_PI, mem::ManuallyDrop};

use num_complex::Complex64;
use quadrature::double_exponential;

#[derive(Default)]
pub struct Heston {
  /// Initial stock price
  pub s0: f64,
  /// Initial volatility
  pub v0: f64,
  /// Strike price
  pub k: f64,
  /// Risk-free rate
  pub r: f64,
  /// Dividend yield
  pub q: f64,
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
  pub tau: Option<ValueOrVec<f64>>,
  /// Evaluation date
  pub eval: Option<ValueOrVec<chrono::NaiveDate>>,
  /// Expiration date
  pub expiry: Option<ValueOrVec<chrono::NaiveDate>>,
}

pub union ValueOrVec<T>
where
  T: Copy,
{
  pub x: T,
  pub v: ManuallyDrop<Vec<T>>,
}

impl Clone for ValueOrVec<f64> {
  fn clone(&self) -> Self {
    unsafe {
      Self {
        v: ManuallyDrop::new(self.v.clone().to_vec()),
      }
    }
  }
}

impl Clone for ValueOrVec<chrono::NaiveDate> {
  fn clone(&self) -> Self {
    unsafe {
      Self {
        v: ManuallyDrop::new(self.v.clone().to_vec()),
      }
    }
  }
}

impl Heston {
  /// Create a new Heston model
  #[must_use]
  pub fn new(params: &Self) -> Self {
    Self {
      s0: params.s0,
      v0: params.v0,
      k: params.k,
      r: params.r,
      q: params.q,
      rho: params.rho,
      kappa: params.kappa,
      theta: params.theta,
      sigma: params.sigma,
      lambda: params.lambda,
      tau: params.tau.clone(),
      eval: params.eval.clone(),
      expiry: params.expiry.clone(),
    }
  }

  /// Calculate the price of a European call option using the Heston model
  /// https://quant.stackexchange.com/a/18686
  pub fn price(&self) -> ValueOrVec<(f64, f64)> {
    if self.tau.is_none() && self.eval.is_none() && self.expiry.is_none() {
      panic!("At least 2 of tau, eval, and expiry must be provided");
    }

    let lambda = self.lambda.unwrap_or(0.0);

    let u = |j: u8| match j {
      1 => 0.5,
      2 => -0.5,
      _ => panic!("Invalid j"),
    };

    let b = |j: u8| match j {
      1 => self.kappa + lambda,
      2 => self.kappa + lambda - self.rho * self.sigma,
      _ => panic!("Invalid j"),
    };

    let d = |j: u8, phi: f64| -> Complex64 {
      ((b(j) - self.rho * self.sigma * phi * Complex64::i()).powi(2)
        - self.sigma.powi(2) * (2.0 * Complex64::i() * u(j) * phi - phi.powi(2)))
      .sqrt()
    };

    let g = |j: u8, phi: f64| -> Complex64 {
      (b(j) - self.rho * self.sigma * Complex64::i() * phi + d(j, phi))
        / (b(j) - self.rho * self.sigma * Complex64::i() * phi - d(j, phi))
    };

    let C = |j: u8, phi: f64, tau: f64| -> Complex64 {
      (self.r - self.q) * Complex64::i() * phi * tau
        + (self.kappa * self.theta / self.sigma.powi(2))
          * ((b(j) - self.rho * self.sigma * Complex64::i() * phi + d(j, phi)) * tau
            - 2.0 * ((1.0 - g(j, phi) * (d(j, phi) * tau).exp()) / (1.0 - g(j, phi))).ln())
    };

    let D = |j: u8, phi: f64, tau: f64| -> Complex64 {
      ((b(j) - self.rho * self.sigma * Complex64::i() * phi + d(j, phi)) / self.sigma.powi(2))
        * ((1.0 - (d(j, phi) * tau).exp()) / (1.0 - g(j, phi) * (d(j, phi) * tau).exp()))
    };

    let f = |j: u8, phi: f64, tau: f64| -> Complex64 {
      (C(j, phi, tau) + D(j, phi, tau) * self.v0 + Complex64::i() * phi * self.s0.ln()).exp()
    };

    unsafe {
      let tau = self.tau.as_ref().unwrap();

      if tau.v.is_empty() {
        let tau = tau.x;

        let re1 =
          |phi: f64| -> f64 { (f(1, phi, tau) * (-Complex64::i() * phi * self.k.ln()).exp()).re };
        let re2 =
          |phi: f64| -> f64 { (f(1, phi, tau) * (-Complex64::i() * phi * self.k.ln()).exp()).re };

        let p1 =
          0.5 + FRAC_1_PI * double_exponential::integrate(re1, 0.000001, 50.0, 10e-6).integral;
        let p2 =
          0.5 + FRAC_1_PI * double_exponential::integrate(re2, 0.000001, 50.0, 10e-6).integral;

        let call = self.s0 * (-self.q * tau).exp() * p1 - self.k * (-self.r * tau).exp() * p2;
        let put = call + self.k * (-self.r * tau).exp() - self.s0 * (-self.q * tau).exp();

        ValueOrVec { x: (call, put) }
      } else {
        let mut prices = Vec::with_capacity(tau.v.len());

        for tau in tau.v.iter() {
          let re1 = |phi: f64| -> f64 {
            (f(1, phi, *tau) * (-Complex64::i() * phi * self.k.ln()).exp()).re
          };
          let re2 = |phi: f64| -> f64 {
            (f(1, phi, *tau) * (-Complex64::i() * phi * self.k.ln()).exp()).re
          };

          let p1 =
            0.5 + FRAC_1_PI * double_exponential::integrate(re1, 0.000001, 50.0, 10e-6).integral;
          let p2 =
            0.5 + FRAC_1_PI * double_exponential::integrate(re2, 0.000001, 50.0, 10e-6).integral;

          let call = self.s0 * (-self.q * tau).exp() * p1 - self.k * (-self.r * tau).exp() * p2;
          let put = call + self.k * (-self.r * tau).exp() - self.s0 * (-self.q * tau).exp();

          prices.push((call, put));
        }

        ValueOrVec {
          v: ManuallyDrop::new(prices),
        }
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_price_single_tau() {
    let heston = Heston {
      s0: 100.0,
      v0: 0.04,
      k: 100.0,
      r: 0.05,
      q: 0.02,
      rho: -0.7,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      lambda: Some(0.0),
      tau: Some(ValueOrVec { x: 1.0 }), // Single f64 tau value
      eval: None,
      expiry: None,
    };

    let price = heston.price();

    unsafe {
      match price {
        ValueOrVec { x: (call, put) } => {
          println!("Call Price: {}, Put Price: {}", call, put);
          assert!(call > 0.0);
          assert!(put > 0.0);
        }
      }
    }
  }

  #[test]
  fn test_price_vec_tau() {
    let heston = Heston {
      s0: 100.0,
      v0: 0.04,
      k: 100.0,
      r: 0.05,
      q: 0.02,
      rho: -0.7,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      lambda: Some(0.0),
      tau: Some(ValueOrVec {
        v: ManuallyDrop::new(vec![1.0, 2.0, 3.0]),
      }), // Vec<f64> tau
      eval: None,
      expiry: None,
    };

    let price = heston.price();

    unsafe {
      match price {
        ValueOrVec { v } => {
          for (i, &(call, put)) in v.iter().enumerate() {
            println!(
              "Time to maturity {}: Call Price: {}, Put Price: {}",
              i + 1,
              call,
              put
            );
            assert!(call > 0.0);
            assert!(put > 0.0);
          }
        }
      }
    }
  }
}
