use linreg::linear_regression;

/// Calculates the fractal dimension of a time series using the Higuchi method.
///
/// The Higuchi method is a popular technique for estimating the fractal dimension of a time series,
/// which can be used to analyze the complexity and self-similarity of the data.
///
/// # Parameters
///
/// - `x`: A slice of `f64` representing the time series data.
/// - `kmax`: The maximum value of `k` to be used in the calculation.
///
/// # Returns
///
/// A `f64` value representing the estimated fractal dimension of the time series.
///
/// # Example
///
/// ```
/// use stochastic_rs::statistics::higuchi_fd::higuchi_fd;
///
/// let data = vec![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0];
/// let fd = higuchi_fd(&data, 5);
/// println!("Fractal Dimension: {}", fd);
/// ```
///
/// # Panics
///
/// This function will panic if the input slice `x` is empty.
pub fn higuchi_fd(x: &[f64], kmax: usize) -> f64 {
  let n_times = x.len();

  let mut lk = vec![0.0; kmax];
  let mut x_reg = vec![0.0; kmax];
  let mut y_reg = vec![0.0; kmax];

  for k in 1..=kmax {
    let mut lm = vec![0.0; k];

    for m in 0..k {
      let mut ll = 0.0;
      let n_max = ((n_times - m - 1) as f64 / k as f64).floor() as usize;

      for j in 1..n_max {
        ll += (x[m + j * k] - x[m + (j - 1) * k]).abs();
      }

      ll /= k as f64;
      ll *= (n_times - 1) as f64 / (k * n_max) as f64;
      lm[m] = ll;
    }

    lk[k - 1] = lm.iter().sum::<f64>() / k as f64;
    x_reg[k - 1] = (1.0 / k as f64).ln();
    y_reg[k - 1] = lk[k - 1].ln();
  }

  let (slope, _) = linear_regression(&x_reg, &y_reg).unwrap();
  slope
}
