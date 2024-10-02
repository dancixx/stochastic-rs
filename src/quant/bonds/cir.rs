use crate::quant::r#trait::Price;

/// CIR model for zero-coupon bond pricing
/// dR(t) = theta(mu - R(t))dt + sigma * sqrt(R(t))dW(t)
/// where R(t) is the short rate.
#[derive(Default, Debug)]
pub struct CIR {
  /// Initial short rate
  pub r_t: f64,
  /// Long-term mean of the short rate
  pub theta: f64,
  /// Mean reversion speed
  pub mu: f64,
  /// Volatility
  pub sigma: f64,
  /// Maturity of the bond in days
  pub tau: f64,
  /// Evaluation date
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date
  pub expiration: Option<chrono::NaiveDate>,
}

impl Price for CIR {
  fn price(&self) -> f64 {
    todo!()
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
