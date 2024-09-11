use linreg::linear_regression;

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
