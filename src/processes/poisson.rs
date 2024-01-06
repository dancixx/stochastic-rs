use ndarray::Array1;
use rand::{thread_rng, Rng};

pub fn poisson(n: usize, lambda: usize) -> [Vec<f64>; 3] {
  if lambda == 0 || n == 0 {
    panic!("Lambda and n must be positive integers");
  }

  let mut times = Array1::<f64>::zeros(n);
  let mut times_total = Array1::<f64>::zeros(n + 1);
  let mut x = Array1::<f64>::zeros(n + 1);

  for i in 1..n {
    times[i] = -(1.0 / lambda as f64) * (lambda as f64).ln();
    times_total[i] = times_total[i - 1] + times[i];
    x[i] = x[i - 1] + 1.0;
  }

  [times.to_vec(), times_total.to_vec(), x.to_vec()]
}

pub fn compound_poisson(n: usize, lambda: usize, jumps: Option<Vec<f64>>) -> [Vec<f64>; 3] {
  if lambda == 0 || n == 0 {
    panic!("Lambda and n must be positive integers");
  }

  let [times, times_total, x] = poisson(n, lambda);

  let mut y = Array1::<f64>::zeros(x.len() + 1);
  let jumps = match jumps {
    Some(jumps) => jumps,
    None => thread_rng()
      .sample_iter(rand_distr::StandardNormal)
      .take(n)
      .collect(),
  };

  for i in 1..n {
    y[i] = y[i - 1] + jumps[i];
  }

  [times.to_vec(), times_total.to_vec(), y.to_vec()]
}
