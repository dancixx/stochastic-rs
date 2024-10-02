use crate::quant::r#trait::Price;

/// Hull-White model for zero-coupon bond pricing
/// dR(t) = (theta(t) - aR(t))dt + sigma(t)dW(t)
/// where R(t) is the short rate.
#[derive(Debug)]
pub struct HullWhite {
  /// Short rate
  pub r_t: f64,
  /// Long-term mean of the short rate
  pub theta: fn(f64) -> f64,
  /// Mean reversion speed
  pub alpha: f64,
  /// Volatility
  pub sigma: f64,
  /// Maturity of the bond in days
  pub tau: f64,
  /// Evaluation date
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date
  pub expiration: Option<chrono::NaiveDate>,
}

impl Price for HullWhite {
  fn price(&self) -> f64 {
    0.0
  }

  fn tau(&self) -> Option<f64> {
    Some(self.tau)
  }

  fn eval(&self) -> Option<chrono::NaiveDate> {
    self.eval
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    self.expiration
  }
}
