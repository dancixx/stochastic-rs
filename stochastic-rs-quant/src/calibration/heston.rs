use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{storage, DMatrix, DVector, Dyn, OVector};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use stochastic_rs::volatility::heston::Heston;

use crate::pricer::heston::HestonPricer;

struct Calibrator<'a> {
  p: DVector<f64>,
  pricer: &'a HestonPricer,
}

impl<'a> Calibrator<'a> {
  fn new(p: OVector<f64, Dyn>, pricer: &'a HestonPricer) -> Self {
    Self { p, pricer }
  }
}

impl<'a> LeastSquaresProblem<f64, Dyn, Dyn> for Calibrator<'a> {
  type JacobianStorage = storage::Owned<f64, Dyn, Dyn>;
  type ParameterStorage = storage::Owned<f64, Dyn>;
  type ResidualStorage = storage::Owned<f64, Dyn>;

  fn set_params(&mut self, p: &DVector<f64>) {
    self.p.copy_from(p);
  }

  fn params(&self) -> DVector<f64> {
    self.p.clone()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let model_prices = self.pricer.prices.as_ref().unwrap();
    let call_prices = unsafe {
      model_prices
        .v
        .clone()
        .iter()
        .map(|x| x.0)
        .collect::<Vec<f64>>()
    };
    // Add some noise to the market prices
    let market_prices = call_prices
      .iter()
      .map(|x| *x + Normal::new(1.0, 0.5).unwrap().sample(&mut thread_rng()))
      .collect::<Vec<f64>>();

    let residuals = call_prices
      .iter()
      .zip(market_prices.iter())
      .map(|(x, y)| x - y)
      .collect::<Vec<f64>>();

    Some(DVector::from_vec(residuals))
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    let derivates = self.pricer.derivates.as_ref().unwrap();
    let derivates = unsafe { derivates.v.clone().to_vec() };

    // Convert flattened vector to a matrix
    let jacobian = DMatrix::from_vec(derivates.len() / 5, 5, derivates);
    Some(jacobian)
  }
}

pub struct HestonCalibrator {
  //model: ,
  pricer: HestonPricer,
}

impl HestonCalibrator {
  #[must_use]
  pub fn new(pricer: HestonPricer) -> Self {
    Self { pricer }
  }

  pub fn calibrate(&mut self) {
    self.pricer.price();
    let (result, report) = LevenbergMarquardt::new().minimize(Calibrator::new(
      DVector::from_vec(vec![0.05, 0.05, -0.8, 5.0, 0.5]),
      &self.pricer,
    ));
    println!("{:?}", result.p);
    println!("{:?}", report.number_of_evaluations);
    println!("{:?}", report.termination);
  }
}

#[cfg(test)]
mod tests {
  use std::mem::ManuallyDrop;

  use crate::ValueOrVec;

  use super::*;

  #[test]
  fn test_calibrate() {
    let majurities = (0..=100)
      .map(|x| 0.5 + 0.1 * x as f64)
      .collect::<Vec<f64>>();
    let mut calibrator = HestonCalibrator::new(HestonPricer {
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
      tau: Some(ValueOrVec {
        v: ManuallyDrop::new(majurities.clone()),
      }), // Single f64 tau value
      eval: None,
      expiry: None,
      prices: None,
      derivates: None,
    });
    calibrator.calibrate();
  }
}
