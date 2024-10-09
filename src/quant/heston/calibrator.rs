use std::cell::RefCell;

use impl_new_derive::ImplNew;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector, Dyn, Owned};
use ndarray::{Array, Array1};

use crate::{
  quant::{r#trait::Pricer, OptionType},
  stats::mle::nmle_heston,
};

use super::pricer::HestonPricer;

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

    // Print the c_market
    println!("Market prices: {:?}", self.c_market);

    // Print the c_model
    println!("Model prices: {:?}", result.residuals().unwrap());

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

    for (idx, _) in self.c_market.iter().enumerate() {
      let pricer = HestonPricer::new(
        self.s[idx],
        self.params[0],
        self.k[idx],
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
      let (call, put) = pricer.calculate_price();

      match self.option_type {
        OptionType::Call => c_model[idx] = call,
        OptionType::Put => c_model[idx] = put,
      }

      derivates.push(pricer.derivatives());
    }

    let _ = std::mem::replace(&mut *self.derivates.borrow_mut(), derivates);
    // println!("c_model: {:?}", c_model);
    // println!("c_market: {:?}", self.c_market.clone());
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

#[cfg(test)]
mod tests {

  use polars::series::ChunkCompare;

  use crate::{quant::yahoo::Yahoo, stochastic::N};

  use super::*;

  #[test]
  fn test_heston_calibrate() {
    let mut yahoo = Yahoo::default();
    yahoo.set_symbol("AAPL");
    yahoo.get_options_chain(&OptionType::Call);
    yahoo.get_price_history();
    let options = yahoo.options.as_ref().unwrap();
    // remove where last price less than 0.01
    let mask = options.column("last_price").unwrap().gt(10.00).unwrap();
    let options = options.filter(&mask).unwrap();

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
      s.clone(),
      k.clone(),
      0.5,
      None,
      c_market,
      0.5,
      OptionType::Call,
      None,
    );
    calibrator.initial_params(Array1::from(s), Array1::default(N), 0.05);
    calibrator.calibrate();
  }
}
