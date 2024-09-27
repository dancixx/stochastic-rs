use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{storage, DMatrix, DVector, Dyn, OMatrix, OVector};
use stochastic_rs::{volatility::heston::Heston, Sampling2D};

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
    let model_prices = vec![100.0];
    let market_prices = vec![89.0];

    let residuals = model_prices
      .iter()
      .zip(market_prices.iter())
      .map(|(m, p)| m - p)
      .collect::<Vec<f64>>();

    Some(DVector::from_vec(residuals))
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    let dC_dv0 = self.pricer.dC_dv0();
    let dC_dtheta = self.pricer.dC_dtheta();
    let dC_drho = self.pricer.dC_drho();
    let dC_dkappa = self.pricer.dC_dkappa();
    let dC_dsigma = self.pricer.dC_dsigma();

    let jacobian = DMatrix::from_vec(1, 5, vec![dC_dv0, dC_dtheta, dC_drho, dC_dkappa, dC_dsigma]);
    Some(jacobian)
  }
}

pub struct HestonCalibrator {
  //model: Heston,
  pricer: HestonPricer,
}

impl HestonCalibrator {
  #[must_use]
  pub fn new(model: Heston, pricer: HestonPricer) -> Self {
    Self { pricer }
  }

  pub fn calibrate(&self) {
    //let [s, v] = self.model.sample();
    let price = self.pricer.price();

    let (_result, report) = LevenbergMarquardt::new().minimize(Calibrator::new(
      DVector::from_vec(vec![0.05, 0.05, -0.8, 5.0, 0.5]),
      &self.pricer,
    ));
    println!("{:?}", report.objective_function);
    println!("{:?}", report.number_of_evaluations);
    println!("{:?}", report.termination);
  }
}

#[cfg(test)]
mod tests {
  use crate::ValueOrVec;

  use super::*;

  #[test]
  fn test_calibrate() {
    let calibrator = HestonCalibrator::new(
      Heston::default(),
      HestonPricer {
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
      },
    );
    calibrator.calibrate();
  }
}
