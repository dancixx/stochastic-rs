use linreg::linear_regression;

pub fn higuchi_fd(x: &[f32], kmax: usize) -> f32 {
  let n_times = x.len();

  let mut lk = vec![0.0; kmax];
  let mut x_reg = vec![0.0; kmax];
  let mut y_reg = vec![0.0; kmax];

  for k in 1..=kmax {
    let mut lm = vec![0.0; k];

    for m in 0..k {
      let mut ll = 0.0;
      let n_max = ((n_times - m - 1) as f32 / k as f32).floor() as usize;

      for j in 1..n_max {
        ll += (x[m + j * k] - x[m + (j - 1) * k]).abs();
      }

      ll /= k as f32;
      ll *= (n_times - 1) as f32 / (k * n_max) as f32;
      lm[m] = ll;
    }

    lk[k - 1] = lm.iter().sum::<f32>() / k as f32;
    x_reg[k - 1] = (1.0 / k as f32).ln();
    y_reg[k - 1] = lk[k - 1].ln();
  }

  let (slope, _) = linear_regression(&x_reg, &y_reg).unwrap();
  slope
}
