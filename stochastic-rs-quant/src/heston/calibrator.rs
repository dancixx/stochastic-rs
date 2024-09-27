use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DVector;

use crate::calibrator::Calibrator;

use super::pricer::HestonPricer;

pub struct HestonCalibrator {
  pricer: HestonPricer,
}

impl HestonCalibrator {
  #[must_use]
  pub fn new(pricer: HestonPricer) -> Self {
    Self { pricer }
  }

  pub fn calibrate(&mut self) {
    self.pricer.price();
    let (result, ..) = LevenbergMarquardt::new().minimize(Calibrator::new(
      DVector::from_vec(vec![0.05, 0.05, -0.8, 5.0, 0.5]),
      None,
      &self.pricer,
    ));
    println!("{:?}", result.p);
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
