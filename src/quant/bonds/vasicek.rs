use crate::quant::r#trait::Price;

/// Vasicek model for zero-coupon bond pricing
/// dR(t) = theta(mu - R(t))dt + sigma dW(t)
/// where R(t) is the short rate.
#[derive(Default, Debug)]
pub struct Vasicek {
  /// Short rate
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

impl Price for Vasicek {
  fn price(&self) -> f64 {
    let tau = self.calculate_tau_in_days();

    let B = (1.0 - (-self.theta * tau).exp()) / self.theta;
    let A = (self.mu - (self.sigma.powi(2) / (2.0 * self.theta.powi(2)))) * (B - tau)
      - (self.sigma.powi(2) / (4.0 * self.theta)) * B.powi(2);

    (A - B * self.r_t).exp()
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
