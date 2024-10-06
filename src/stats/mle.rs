use ndarray::Array1;

/// Maximum likelihood estimation for Heston model
/// http://scis.scichina.com/en/2018/042202.pdf
///
/// # Arguments
/// s: Vec<f64> - stock prices
/// v: Vec<f64> - volatility
/// r: f64 - risk-free rate
///
/// # Returns
/// Vec<f64> - estimated parameters
pub fn nmle_heston(s: Array1<f64>, v: Array1<f64>, r: f64) -> Vec<f64> {
  let n = v.len();
  let delta = 1.0 / n as f64;
  let mut sum = [0.0; 4];

  for i in 1..n {
    // sum of (V_i / V_{i-1}
    sum[0] += v[i] / v[i - 1];

    // sum of V_i
    sum[1] += v[i];

    // sum of V_{i-1}
    sum[2] += v[i - 1];

    // sum of 1 / V_{i-1}
    sum[3] += 1.0 / v[i - 1];
  }

  let beta_hat1 = ((n as f64).powi(-2) * sum[1] * sum[3] - (n as f64).powi(-1) * sum[0])
    / ((n as f64).powi(-2) * sum[2] * sum[3] - 1.0);
  let beta_hat2 =
    ((n as f64).powi(-1) * sum[0] - beta_hat1) / ((1.0 - beta_hat1) * (n as f64).powi(-1) * sum[3]);

  let sum_beta_hat3 = {
    let mut sum = 0.0;
    for i in 1..n {
      sum +=
        (v[i] - v[i - 1] * beta_hat1 - beta_hat2 * (1.0 - beta_hat1).powi(2)) * (1.0 / v[i - 1])
    }
    sum
  };
  let beta_hat3 = (n as f64).powi(-1) * sum_beta_hat3;
  let kappa_hat = -(1.0 / delta) * beta_hat1.ln();
  let theta_hat = beta_hat2;
  let sigma_hat = (2.0 * kappa_hat * beta_hat3) / (1.0 - beta_hat1.powi(2)).sqrt();

  let mut sum_dw1dw2 = 0.0;
  for i in 1..n {
    let dw1_i = (s[i].ln() - s[i - 1].ln() - (r - 0.5 * v[i - 1]) * delta) / v[i - 1].sqrt();
    let dw2_i = (v[i] - v[i - 1] - kappa_hat * (theta_hat - v[i - 1]) * delta)
      / (sigma_hat * v[i - 1].sqrt());

    sum_dw1dw2 += dw1_i * dw2_i;
  }

  let rho_hat = sum_dw1dw2 / (n as f64 * delta);

  vec![
    *v.first().unwrap(),
    theta_hat,
    rho_hat,
    kappa_hat,
    sigma_hat,
  ]
}
