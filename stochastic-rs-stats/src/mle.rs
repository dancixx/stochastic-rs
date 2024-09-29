/// Maximum likelihood estimation for Heston model
///
/// # Arguments
/// s: Vec<f64> - stock prices
/// v: Vec<f64> - volatility
/// r: f64 - risk-free rate
///
/// # Returns
/// Vec<f64> - estimated parameters
pub fn mle_heston(s: Vec<f64>, v: Vec<f64>, r: f64) -> Vec<f64> {
  let n = v.len();
  let delta = 1.0 / n as f64;
  let mut sum = [0.0; 6];

  for i in 1..n {
    // sum of sqrt(V_i * V_{i-1})
    sum[0] += (v[i] * v[i - 1]).sqrt();

    // sum of sqrt(V_i / V_{i-1})
    sum[1] += (v[i] / v[i - 1]).sqrt();

    // sum of V_i
    sum[2] += v[i];

    // sum of V_{i-1}
    sum[3] += v[i - 1];

    // sum of sqrt(V_i)
    sum[4] += v[i].sqrt();

    // sum of sqrt(V_{i-1})
    sum[5] += v[i - 1].sqrt();
  }

  let P_hat = ((1.0 / n as f64) * sum[0] - (1.0 / n as f64).powi(2) * sum[1] * sum[3])
    / ((delta / 2.0) - (delta / 2.0) * (1.0 / n as f64).powi(2) * (1.0 / sum[3]) * sum[3]);

  let kappa_hat = (2.0 / delta)
    * (1.0 + (P_hat * delta / 2.0) * (1.0 / n as f64) * (1.0 / sum[3]) - (1.0 / n as f64) * sum[1]);

  let sigma_hat = ((4.0 / delta)
    * (1.0 / n as f64)
    * (sum[4] - sum[5] - (delta / (2.0 * sum[5])) * (P_hat - kappa_hat * sum[3])).powi(2))
  .sqrt();

  let theta_hat = (P_hat + 0.25 * sigma_hat.powi(2)) / kappa_hat;

  let mut sum_dw1dw2 = 0.0;
  for i in 1..n {
    let dw1_i = (s[i].ln() - s[i - 1].ln() - (r - 0.5 * v[i - 1]) * delta) / v[i - 1].sqrt();
    let dw2_i = (v[i] - v[i - 1] - kappa_hat * (theta_hat - v[i - 1]) * delta)
      / (sigma_hat * v[i - 1].sqrt());

    sum_dw1dw2 += dw1_i * dw2_i;
  }

  let rho_hat = sum_dw1dw2 / (n as f64 * delta);

  vec![
    v.first().unwrap().clone(),
    theta_hat,
    rho_hat,
    kappa_hat,
    sigma_hat,
  ]
}
