use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DVector;

use crate::{calibrator::Calibrator, yahoo::Yahoo};

use super::pricer::HestonPricer;

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
    let delta = 1.0 / 252.0;
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

    println!("{:?}", impl_vol);
    let n = impl_vol.len();
    let mut sum = [0.0; 6];

    for i in 0..n {
      if i > 0 {
        sum[0] += (impl_vol[i] * impl_vol[i - 1]).sqrt();
        sum[1] += (impl_vol[i] / impl_vol[i - 1]).sqrt();

        sum[3] += impl_vol[i - 1];
        sum[5] += impl_vol[i - 1].sqrt();
      }
      sum[2] += impl_vol[i];
      sum[4] += impl_vol[i].sqrt();
    }

    let P_hat = ((1.0 / n as f64) * sum[0] - (1.0 / n as f64).powi(2) * sum[1] * sum[2])
      / (delta - (delta / 2.0) * (1.0 / n as f64).powi(2) * (1.0 / sum[3]) * sum[2]);

    let kappa_hat = (2.0 / delta)
      * (1.0 + (P_hat * delta / 2.0) * (1.0 / n as f64) * (1.0 / sum[3])
        - (1.0 / n as f64) * sum[1]);

    let sigma_hat = ((4.0 / delta)
      * (1.0 / n as f64)
      * (sum[4] - sum[5] - (delta / (2.0 * sum[5])) * (P_hat - kappa_hat * sum[3])).powi(2))
    .sqrt();

    let theta_hat = (P_hat - 0.25 * sigma_hat.powi(2)) / kappa_hat;

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
  use std::mem::ManuallyDrop;

  use stochastic_rs::{volatility::heston::Heston, Sampling2D};

  use crate::ValueOrVec;

  use super::*;

  #[test]
  fn test_calibrate() {
    let majurities = (0..=100)
      .map(|x| 0.5 + 0.1 * x as f64)
      .collect::<Vec<f64>>();
    let heston = Heston::new(&Heston {
      s0: Some(100.0),
      v0: Some(0.05),
      rho: -0.3,
      kappa: 2.0,
      theta: 0.05,
      sigma: 0.5,
      mu: 2.0,
      n: 252,
      t: Some(1.0),
      use_sym: Some(false),
      m: None,
      cgns: Default::default(),
    });
    let data = heston.sample();
    let mut calibrator = HestonCalibrator::new(
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
        tau: Some(ValueOrVec {
          v: ManuallyDrop::new(majurities.clone()),
        }), // Single f64 tau value
        eval: None,
        expiry: None,
        prices: None,
        derivates: None,
      },
      Yahoo::default(),
      Some(data[0].to_vec()),
      Some(data[1].to_vec()),
    );
    let guess = calibrator.initial_guess();
    println!("{:?}", guess);
    //calibrator.calibrate();
  }
}
