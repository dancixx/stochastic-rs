use std::cell::RefCell;

use impl_new_derive::ImplNew;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector, Dyn, Owned};
use ndarray::Array1;

use crate::{
  quant::{r#trait::Pricer, OptionType},
  stats::mle::nmle_heston,
};

use super::pricer::HestonPricer;

/// Heston model parameters
#[derive(Clone, Debug)]
pub struct HestonParams {
  pub v0: f64,
  pub theta: f64,
  pub rho: f64,
  pub kappa: f64,
  pub sigma: f64,
}

impl From<HestonParams> for DVector<f64> {
  fn from(params: HestonParams) -> Self {
    DVector::from_vec(vec![
      params.v0,
      params.theta,
      params.rho,
      params.kappa,
      params.sigma,
    ])
  }
}

impl From<DVector<f64>> for HestonParams {
  fn from(params: DVector<f64>) -> Self {
    HestonParams {
      v0: params[0],
      theta: params[1],
      rho: params[2],
      kappa: params[3],
      sigma: params[4],
    }
  }
}

/// Heston calibrator
#[derive(ImplNew)]
pub struct HestonCalibrator {
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
  pub initial_params: Option<HestonParams>,
}

impl HestonCalibrator {
  pub fn calibrate(&mut self) {
    println!("Initial guess: {:?}", self.initial_params.as_ref().unwrap());

    let (result, report) = LevenbergMarquardt::new().minimize(HestonCalibrationProblem::new(
      self.initial_params.as_ref().unwrap().clone(),
      self.c_market.clone().into(),
      self.s.clone().into(),
      self.k.clone().into(),
      self.tau,
      self.r,
      self.q,
      &self.option_type,
    ));

    // Print the c_market
    println!("Market prices: {:?}", self.c_market);

    let residuals = result.residuals().unwrap();

    // Print the c_model
    println!(
      "Model prices: {:?}",
      DVector::from_vec(self.c_market.clone()) + residuals
    );

    // Print the result of the calibration
    println!("Calibration report: {:?}", result.params);
  }

  /// Initial guess for the calibration
  /// http://scis.scichina.com/en/2018/042202.pdf
  ///
  /// Using NMLE (Normal Maximum Likelihood Estimation) method
  pub fn initial_params(&mut self, s: Array1<f64>, v: Array1<f64>, r: f64) {
    self.initial_params = Some(nmle_heston(s, v, r));
  }
}

/// A calibrator.
#[derive(ImplNew)]
pub(crate) struct HestonCalibrationProblem<'a> {
  /// Params to calibrate.
  pub params: HestonParams,
  /// Option prices from the market.
  pub c_market: DVector<f64>,
  /// Asset price vector.
  pub s: DVector<f64>,
  /// Strike price vector.
  pub k: DVector<f64>,
  /// Time to maturity.
  pub tau: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: Option<f64>,
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
    self.params = HestonParams::from(params.clone());
  }

  fn params(&self) -> DVector<f64> {
    self.params.clone().into()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let mut c_model = DVector::zeros(self.c_market.len());
    let mut derivates = Vec::new();

    for (idx, _) in self.c_market.iter().enumerate() {
      let pricer = HestonPricer::new(
        self.s[idx],
        self.params.v0,
        self.k[idx],
        self.r,
        self.q,
        self.params.rho,
        self.params.kappa,
        self.params.theta,
        self.params.sigma,
        None,
        self.tau,
        None,
        None,
      );
      let (call, put) = pricer.calculate_price();

      match self.option_type {
        OptionType::Call => c_model[idx] = call,
        OptionType::Put => c_model[idx] = put,
      }

      derivates.push(pricer.derivatives());
    }

    let _ = std::mem::replace(&mut *self.derivates.borrow_mut(), derivates);
    Some(c_model - self.c_market.clone())
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    let derivates = self.derivates.borrow();
    let derivates = derivates.iter().flatten().cloned().collect::<Vec<f64>>();

    // The Jacobian matrix is a matrix of partial derivatives
    // of the residuals with respect to the parameters.
    let jacobian = DMatrix::from_vec(derivates.len() / 5, 5, derivates);

    Some(jacobian)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_heston_calibrate() {
    let tau = 24.0 / 365.0;
    println!("Time to maturity: {}", tau);

    let s = vec![
      425.73, 425.73, 425.73, 425.67, 425.68, 425.65, 425.65, 425.68, 425.65, 425.16, 424.78,
      425.19,
    ];

    let k = vec![
      395.0, 400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0, 440.0, 445.0, 450.0,
    ];

    let c_market = vec![
      30.75, 25.88, 21.00, 16.50, 11.88, 7.69, 4.44, 2.10, 0.78, 0.25, 0.10, 0.10,
    ];

    let v0 = Array1::linspace(0.0, 0.01, 10);

    for v in v0.iter() {
      let mut calibrator = HestonCalibrator::new(
        s.clone(),
        k.clone(),
        6.40e-4,
        None,
        c_market.clone(),
        tau,
        OptionType::Call,
        Some(HestonParams {
          v0: *v,
          theta: 6.47e-5,
          rho: -1.98e-3,
          kappa: 6.57e-3,
          sigma: 5.09e-4,
        }),
      );
      calibrator.calibrate();
    }
  }
}
