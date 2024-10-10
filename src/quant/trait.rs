use chrono::Local;
use nalgebra::DVector;

/// Pricer trait.
pub(crate) trait Pricer {
  /// Calculate the price of an option.
  fn calculate_call_put(&self) -> (f64, f64) {
    todo!()
  }

  /// Calculate the price
  fn calculate_price(&self) -> f64 {
    todo!()
  }

  /// Update the parameters.
  fn update_params(&mut self, params: DVector<f64>) {
    todo!()
  }
  /// Update strike price.
  fn update_strike(&mut self, k: f64) {
    todo!()
  }
  /// Derivatives.
  fn derivatives(&self) -> Vec<f64> {
    todo!()
  }

  /// Calculate the valuation date of an instrument.
  fn calculate_tau_in_days(&self) -> f64 {
    if let Some(tau) = self.tau() {
      tau
    } else {
      let eval = self
        .eval()
        .unwrap_or_else(|| Local::now().naive_local().into());
      let expiration = self.expiration().unwrap();
      (expiration - eval).num_days() as f64
    }
  }

  /// Calculate the valuation date of an instrument.
  fn calculate_tau_in_years(&self) -> f64 {
    if let Some(tau) = self.tau() {
      tau
    } else {
      let eval = self
        .eval()
        .unwrap_or_else(|| Local::now().naive_local().into());
      let expiration = self.expiration().unwrap();
      (expiration - eval).num_days() as f64 / 365.0
    }
  }

  fn tau(&self) -> Option<f64>;
  fn eval(&self) -> Option<chrono::NaiveDate>;
  fn expiration(&self) -> Option<chrono::NaiveDate>;

  /// Calculate the implied volatility using the Newton-Raphson method.
  fn implied_volatility(&self, _c_market: f64) -> f64 {
    todo!()
  }
}
