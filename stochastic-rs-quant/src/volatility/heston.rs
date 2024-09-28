use std::{f64::consts::FRAC_1_PI, mem::ManuallyDrop};

use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DVector;
use num_complex::Complex64;
use quadrature::double_exponential;

use crate::{volatility::Calibrator, yahoo::Yahoo, ValueOrVec};

use super::Pricer;

#[derive(Default, Clone)]
pub struct HestonPricer {
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
  /// Prices of European call and put options
  pub(crate) prices: Option<ValueOrVec<(f64, f64)>>,
  /// Partial derivative of the C function with respect to the parameters
  pub(crate) derivates: Option<ValueOrVec<f64>>,
}

impl Pricer for HestonPricer {
  /// Prices.
  fn prices(&self) -> Option<ValueOrVec<(f64, f64)>> {
    self.prices.clone()
  }

  /// Derivatives.
  fn derivates(&self) -> Option<ValueOrVec<f64>> {
    self.derivates.clone()
  }
}

impl HestonPricer {
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
      lambda: Some(params.lambda.unwrap_or(0.0)),
      tau: params.tau.clone(),
      eval: params.eval.clone(),
      expiry: params.expiry.clone(),
      prices: None,
      derivates: None,
    }
  }

  /// Calculate the price of a European call option using the Heston model
  /// https://quant.stackexchange.com/a/18686
  pub fn price(&mut self) -> ValueOrVec<(f64, f64)> {
    if self.tau.is_none() && self.eval.is_none() && self.expiry.is_none() {
      panic!("At least 2 of tau, eval, and expiry must be provided");
    }

    unsafe {
      let tau = self.tau.as_ref().unwrap();

      if tau.v.is_empty() {
        let tau = tau.x;

        let call = self.s0 * (-self.q * tau).exp() * self.p(1, tau)
          - self.k * (-self.r * tau).exp() * self.p(2, tau);
        let put = call + self.k * (-self.r * tau).exp() - self.s0 * (-self.q * tau).exp();

        self.prices = Some(ValueOrVec { x: (call, put) });
        self.derivates = Some(ValueOrVec {
          v: ManuallyDrop::new(self.derivates(tau)),
        });
        ValueOrVec { x: (call, put) }
      } else {
        let mut prices = Vec::with_capacity(tau.v.len());
        let mut derivatives = Vec::with_capacity(tau.v.len());

        for tau in tau.v.iter() {
          let call = self.s0 * (-self.q * tau).exp() * self.p(1, *tau)
            - self.k * (-self.r * tau).exp() * self.p(2, *tau);
          let put = call + self.k * (-self.r * tau).exp() - self.s0 * (-self.q * tau).exp();

          prices.push((call, put));
          derivatives.push(self.derivates(*tau));
        }

        self.prices = Some(ValueOrVec {
          v: ManuallyDrop::new(prices.clone()),
        });

        // Flatten the derivatives vector
        self.derivates = Some(ValueOrVec {
          v: ManuallyDrop::new(derivatives.into_iter().flatten().collect::<Vec<f64>>()),
        });

        ValueOrVec {
          v: ManuallyDrop::new(prices),
        }
      }
    }
  }

  pub(self) fn u(&self, j: u8) -> f64 {
    match j {
      1 => 0.5,
      2 => -0.5,
      _ => panic!("Invalid j"),
    }
  }

  pub(self) fn b(&self, j: u8) -> f64 {
    match j {
      1 => self.kappa + self.lambda.unwrap() - self.rho * self.sigma,
      2 => self.kappa + self.lambda.unwrap(),
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
    (self.r - self.q) * Complex64::i() * phi * tau
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
    (self.C(j, phi, tau) + self.D(j, phi, tau) * self.v0 + Complex64::i() * phi * self.s0.ln())
      .exp()
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

  pub(crate) fn derivates(&self, tau: f64) -> Vec<f64> {
    vec![
      self.dC_dv0(tau),
      self.dC_dtheta(tau),
      self.dC_drho(tau),
      self.dC_dkappa(tau),
      self.dC_dsigma(tau),
    ]
  }
}

/// Heston calibrator
pub struct HestonCalibrator<'a> {
  /// Yahoo struct
  pub yahoo: Yahoo<'a>,
  /// Implied volatility vector
  pub v: Option<Vec<f64>>,
  /// Prices vector
  pub p: Option<Vec<f64>>,
  /// Heston pricer
  pricer: HestonPricer,
}

impl<'a> HestonCalibrator<'a> {
  #[must_use]
  pub fn new(
    pricer: HestonPricer,
    yahoo: Yahoo<'a>,
    p: Option<Vec<f64>>,
    v: Option<Vec<f64>>,
  ) -> Self {
    Self {
      pricer,
      yahoo,
      p,
      v,
    }
  }

  pub fn calibrate(&mut self) {
    self.yahoo.get_options_chain();
    self.pricer.price();
    let (result, ..) = LevenbergMarquardt::new().minimize(Calibrator::new(
      DVector::from_vec(vec![0.05, 0.05, -0.8, 5.0, 0.5]),
      None,
      &self.pricer,
    ));
    println!("{:?}", result.p);
  }

  /// Initial guess for the calibration
  /// http://scis.scichina.com/en/2018/042202.pdf
  /// Returns [v0, theta, rho, kappa, sigma]
  fn initial_guess(&self) -> DVector<f64> {
    let impl_vol = if let Some(v) = &self.v {
      v.to_owned()
    } else {
      let options = self.yahoo.options.clone().unwrap();
      // get impl_volatities col from options
      let impl_vol = options.select(["impl_volatility"]).unwrap();
      // convert to vec
      let impl_vol = impl_vol
        .select_at_idx(0)
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect::<Vec<f64>>();

      impl_vol
    };

    let n = impl_vol.len();
    let delta = 1.0 / n as f64;
    let mut sum = [0.0; 6];

    for i in 1..n {
      // sum of sqrt(V_i * V_{i-1})
      sum[0] += (impl_vol[i] * impl_vol[i - 1]).sqrt();

      // sum of sqrt(V_i / V_{i-1})
      sum[1] += (impl_vol[i] / impl_vol[i - 1]).sqrt();

      // sum of V_i
      sum[2] += impl_vol[i];

      // sum of V_{i-1}
      sum[3] += impl_vol[i - 1];

      // sum of sqrt(V_i)
      sum[4] += impl_vol[i].sqrt();

      // sum of sqrt(V_{i-1})
      sum[5] += impl_vol[i - 1].sqrt();
    }

    let P_hat = ((1.0 / n as f64) * sum[0] - (1.0 / n as f64).powi(2) * sum[1] * sum[3])
      / ((delta / 2.0) - (delta / 2.0) * (1.0 / n as f64).powi(2) * (1.0 / sum[3]) * sum[3]);

    let kappa_hat = (2.0 / delta)
      * (1.0 + (P_hat * delta / 2.0) * (1.0 / n as f64) * (1.0 / sum[3])
        - (1.0 / n as f64) * sum[1]);

    let sigma_hat = ((4.0 / delta)
      * (1.0 / n as f64)
      * (sum[4] - sum[5] - (delta / (2.0 * sum[5])) * (P_hat - kappa_hat * sum[3])).powi(2))
    .sqrt();

    let theta_hat = (P_hat + 0.25 * sigma_hat.powi(2)) / kappa_hat;

    let price = if let Some(p) = &self.p {
      p.to_owned()
    } else {
      let options = self.yahoo.options.clone().unwrap();
      let prices = options.select(["last_price"]).unwrap();
      let prices = prices
        .select_at_idx(0)
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect::<Vec<f64>>();

      prices
    };

    let mut sum_dw1dw2 = 0.0;
    for i in 1..n {
      let dw1_i =
        (price[i].ln() - price[i - 1].ln() - (self.pricer.r - 0.5 * impl_vol[i - 1]) * delta)
          / impl_vol[i - 1].sqrt();
      let dw2_i =
        (impl_vol[i] - impl_vol[i - 1] - kappa_hat * (theta_hat - impl_vol[i - 1]) * delta)
          / (sigma_hat * impl_vol[i - 1].sqrt());

      sum_dw1dw2 += dw1_i * dw2_i;
    }

    let rho_hat = sum_dw1dw2 / (n as f64 * delta);

    DVector::from_vec(vec![
      self.pricer.v0,
      theta_hat,
      rho_hat,
      kappa_hat,
      sigma_hat,
    ])
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs::{volatility::heston::Heston, Sampling2D};

  use super::*;

  #[test]
  fn test_heston_single_price() {
    let mut heston = HestonPricer {
      s0: 100.0,
      v0: 0.05,
      k: 100.0,
      r: 0.03,
      q: 0.02,
      rho: -0.8,
      kappa: 5.0,
      theta: 0.05,
      sigma: 0.5,
      lambda: Some(0.0),
      tau: Some(ValueOrVec { x: 0.5 }), // Single f64 tau value
      eval: None,
      expiry: None,
      prices: None,
      derivates: None,
    };

    let price = heston.price();

    unsafe {
      match price {
        ValueOrVec { x: (call, put) } => {
          println!("Call Price: {}, Put Price: {}", call, put);
        }
      }
    }
  }

  #[test]
  fn test_heston_multi_price() {
    let mut heston = HestonPricer {
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
      prices: None,
      derivates: None,
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
          }
        }
      }
    }
  }

  #[test]
  fn test_heston_calibrate() {
    let majurities = (0..=100)
      .map(|x| 0.5 + 0.1 * x as f64)
      .collect::<Vec<f64>>();
    let heston = Heston::new(&Heston {
      s0: Some(100.0),
      v0: Some(0.2),
      rho: -0.8,
      kappa: 1.0,
      theta: 0.25,
      sigma: 0.5,
      mu: 2.0,
      n: 1000,
      t: Some(1.0),
      use_sym: Some(true),
      m: None,
      cgns: Default::default(),
    });

    let mut data = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    let m = 500;
    for _ in 0..m {
      let heston = heston.sample();
      let calibrator = HestonCalibrator::new(
        HestonPricer {
          s0: 100.0,
          v0: 0.2,
          k: 100.0,
          r: 0.05,
          q: 0.02,
          rho: -0.8,
          kappa: 1.0,
          theta: 0.25,
          sigma: 0.5,
          lambda: Some(0.0),
          tau: Some(ValueOrVec {
            v: ManuallyDrop::new(majurities.clone()),
          }), // Single f64 tau value
          eval: None,
          expiry: None,
          prices: None,
          derivates: None,
        },
        Yahoo::default(),
        Some(heston[0].to_vec()),
        Some(heston[1].to_vec()),
      );
      let guess = calibrator.initial_guess();
      // [v0, theta, rho, kappa, sigma]
      // println!(
      //   "v0: {}, theta: {}, rho: {}, kappa: {}, sigma: {}",
      //   guess[0], guess[1], guess[2], guess[3], guess[4]
      // );

      data[0] += guess[0];
      data[1] += guess[1];
      data[2] += guess[2];
      data[3] += guess[3];
      data[4] += guess[4];
    }

    println!(
      "v0: {}, theta: {}, rho: {}, kappa: {}, sigma: {}",
      data[0] / m as f64,
      data[1] / m as f64,
      data[2] / m as f64,
      data[3] / m as f64,
      data[4] / m as f64
    );
    //calibrator.calibrate();
  }
}
