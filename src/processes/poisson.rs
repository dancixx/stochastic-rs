use std::ops::Div;

use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::{thread_rng, Rng};

pub fn poisson(n: usize, lambda: usize, t: Option<f64>) -> Vec<f64> {
  if t.unwrap() > 0.0 || n == 0 || lambda == 0 {
    panic!("lambda, t and n must be positive integers");
  }

  if t.is_none() {
    let exponentials =
      Array1::<f64>::random(n, rand_distr::Exp::new(1.0.div(lambda as f64)).unwrap());
    let mut p = Array1::<f64>::zeros(n + 1);
    p[0] = 0.0;

    for i in 1..n + 1 {
      p[i] = p[i - 1] + exponentials[i - 1];
    }

    p.to_vec()
  } else {
    let mut p = vec![0.0];
    let mut _t = 0.0;

    while _t < t.unwrap() {
      let exponential = thread_rng().sample(rand_distr::Exp::new(1.0.div(lambda as f64)).unwrap());
      _t += exponential;
      p.push(_t);
    }

    p
  }
}

pub fn compound_poisson(n: usize, lambda: usize, jumps: Option<Vec<f64>>) -> Vec<f64> {
  if n == 0 {
    panic!("n must be a positive integer");
  }

  let times = poisson(n, lambda, None);
  let mut y = Array1::<f64>::zeros(times.len() + 1);
  let jumps = match jumps {
    Some(jumps) => jumps,
    None => thread_rng()
      .sample_iter(rand_distr::StandardNormal)
      .take(n)
      .collect(),
  };

  for i in 1..n {
    y[i] = y[i - 1] + jumps[i - 1];
  }

  y.to_vec()
}
