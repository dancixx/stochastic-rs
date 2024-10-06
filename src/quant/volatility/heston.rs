use std::{cell::RefCell, f64::consts::FRAC_1_PI};

use impl_new_derive::ImplNew;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector, Dyn, Owned};
use ndarray::Array1;
use num_complex::Complex64;
use quadrature::double_exponential;

use crate::{
  quant::{r#trait::Pricer, OptionType},
  stats::mle::nmle_heston,
};

#[derive(ImplNew, Clone, Default)]
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
  /// Prices of European call and put options
  pub(crate) prices: Option<(f64, f64)>,
  /// Partial derivative of the C function with respect to the parameters
  pub(crate) derivates: Option<Vec<f64>>,
}

impl Pricer for HestonPricer {
  /// Calculate the price of a European call option using the Heston model
  /// https://quant.stackexchange.com/a/18686
  fn calculate_price(&mut self) {
    let call = self.s0 * (-self.q.unwrap_or(0.0) * self.tau).exp() * self.p(1, self.tau)
      - self.k * (-self.r * self.tau).exp() * self.p(2, self.tau);
    let put = call + self.k * (-self.r * self.tau).exp()
      - self.s0 * (-self.q.unwrap_or(0.0) * self.tau).exp();

    self.prices = Some((call, put));
    self.derivates = Some(self.derivates(self.tau));
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

  /// Prices.
  fn prices(&self) -> (f64, f64) {
    self.prices.unwrap()
  }

  /// Derivatives.
  fn derivates(&self) -> Vec<f64> {
    self.derivates.clone().unwrap()
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

/// A calibrator.
#[derive(ImplNew)]
pub(crate) struct HestonCalibrationProblem<'a> {
  /// Params to calibrate.
  pub params: DVector<f64>,
  /// Option prices from the market.
  pub c_market: DVector<f64>,
  /// Asset price vector.
  pub s: DVector<f64>,
  /// Strike price vector.
  pub k: DVector<f64>,
  /// Time to maturity.
  pub tau: f64,
  /// Option type
  pub option_type: &'a OptionType,
  /// Derivate matrix.
  derivates: RefCell<Vec<Vec<f64>>>,
}

impl<'a> LeastSquaresProblem<f64, Dyn, Dyn> for HestonCalibrationProblem<'a> {
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    self.params.copy_from(params);
  }

  fn params(&self) -> DVector<f64> {
    self.params.clone()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let mut c_model = DVector::zeros(self.c_market.len());
    let mut derivates = Vec::new();

    for (idx, k) in self.k.iter().enumerate() {
      let mut pricer = HestonPricer::new(
        if self.s.len() > 0 {
          self.s[idx]
        } else {
          self.s[0]
        },
        self.params[0],
        *k,
        0.5,
        None,
        self.params[2],
        self.params[3],
        self.params[1],
        self.params[4],
        None,
        self.tau,
        None,
        None,
      );
      pricer.calculate_price();

      match self.option_type {
        OptionType::Call => c_model[idx] = pricer.prices().0,
        OptionType::Put => c_model[idx] = pricer.prices().1,
      }

      derivates.push(pricer.derivates(self.tau));
    }

    let _ = std::mem::replace(&mut *self.derivates.borrow_mut(), derivates);
    println!("c_model: {:?}", c_model);
    println!("c_market: {:?}", self.c_market.clone());
    Some(c_model - self.c_market.clone())
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    let derivates = self.derivates.borrow();
    let derivates = derivates.iter().flatten().cloned().collect::<Vec<f64>>();

    // The Jacobian matrix is a matrix of partial derivatives
    // of the residuals with respect to the parameters.
    let jacobian = DMatrix::from_vec(
      derivates.len() / self.params.len(),
      self.params.len(),
      derivates,
    );

    Some(jacobian)
  }
}

/// Heston calibrator
#[derive(ImplNew)]
pub struct HestonCalibrator {
  /// Implied volatility vector
  pub v0: f64,
  /// The underlying asset price
  pub s: Vec<f64>,
  /// Strike price vector
  pub k: Vec<f64>,
  /// Risk-free rate
  pub r: f64,
  /// Dividend yield
  pub q: Option<f64>,
  /// Option prices vector from the market
  pub c_market: Vec<f64>,
  /// Time to maturity
  pub tau: f64,
  /// Option type
  pub option_type: OptionType,
  /// Initial guess for the calibration from the NMLE method
  pub initial_params: Option<DVector<f64>>,
}

impl HestonCalibrator {
  pub fn calibrate(&mut self) {
    if self.initial_params.is_none() {
      panic!("Initial guess for the calibration is required. \n Use the initial_params method to set the initial guess \n or use the initial_params argument in the constructor.");
    }

    println!("Initial guess: {:?}", self.initial_params.as_ref().unwrap());

    let (result, report) = LevenbergMarquardt::new().minimize(HestonCalibrationProblem::new(
      self.initial_params.as_ref().unwrap().clone(),
      self.c_market.clone().into(),
      self.s.clone().into(),
      self.k.clone().into(),
      self.tau,
      &self.option_type,
    ));

    // Print the result of the calibration
    println!("Calibration report: {:?}", result.params);

    // Print the result of the calibration
    println!("Calibration report: {:?}", report);
  }

  /// Initial guess for the calibration
  /// http://scis.scichina.com/en/2018/042202.pdf
  ///
  /// Using NMLE (Normal Maximum Likelihood Estimation) method
  pub fn initial_params(&mut self, s: Array1<f64>, v: Array1<f64>, r: f64) {
    self.initial_params = Some(DVector::from_vec(nmle_heston(s, v, r)));
  }
}

#[cfg(test)]
mod tests {

  use crate::quant::yahoo::Yahoo;

  use super::*;

  #[test]
  fn test_heston_single_price() {
    let mut heston = HestonPricer {
      s0: 100.0,
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

    heston.calculate_price();
    let (call, put) = heston.prices();
    println!("Call Price: {}, Put Price: {}", call, put);
  }

  #[test]
  fn test_heston_multi_price() {
    let taus = vec![0.5, 1.0, 2.0, 3.0];

    for tau in taus {
      let mut heston = HestonPricer {
        s0: 100.0,
        v0: 0.05,
        k: 100.0,
        r: 0.03,
        q: Some(0.02),
        rho: -0.8,
        kappa: 5.0,
        theta: 0.05,
        sigma: 0.5,
        lambda: Some(0.0),
        tau,
        ..Default::default()
      };

      heston.calculate_price();
      let (call, put) = heston.prices();
      println!(
        "Time to maturity {}: Call Price: {}, Put Price: {}",
        tau, call, put
      );
    }
  }

  #[test]
  fn test_heston_calibrate() {
    let mut yahoo = Yahoo::default();
    yahoo.set_symbol("AAPL");
    yahoo.get_options_chain(&OptionType::Call);
    yahoo.get_price_history();
    let options = yahoo.options.as_ref().unwrap();
    println!("{:?}", options);

    // Implied volatility
    let v = options.select(["implied_volatility"]).unwrap();
    let v = v
      .select_at_idx(0)
      .unwrap()
      .f64()
      .unwrap()
      .into_no_null_iter()
      .collect::<Vec<f64>>();

    // Get Price history
    let price_history = yahoo.price_history.as_ref().unwrap();
    let s = price_history.select(["close"]).unwrap();
    let s = s
      .select_at_idx(0)
      .unwrap()
      .f64()
      .unwrap()
      .into_no_null_iter()
      .collect::<Vec<f64>>();

    // convert to years the epoch time
    let tau = (yahoo.options_chain.as_ref().unwrap().option_chain.result[0].options[0]
      .expiration_date as f64
      - chrono::Local::now().timestamp() as f64)
      / 31536000.0;
    println!("Time to maturity: {}", tau);
    let c_market = options.select(["last_price"]).unwrap();
    let c_market = c_market
      .select_at_idx(0)
      .unwrap()
      .f64()
      .unwrap()
      .into_no_null_iter()
      .collect::<Vec<f64>>();

    let k = options.select(["strike"]).unwrap();
    let k = k
      .select_at_idx(0)
      .unwrap()
      .f64()
      .unwrap()
      .into_no_null_iter()
      .collect::<Vec<f64>>();

    let mut calibrator = HestonCalibrator::new(
      v[0],
      s.clone(),
      k.clone(),
      0.5,
      None,
      c_market,
      tau,
      OptionType::Call,
      None,
    );
    calibrator.initial_params(Array1::from(s), Array1::from(v), 0.05);
    calibrator.calibrate();
  }
}
