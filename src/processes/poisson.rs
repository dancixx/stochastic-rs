use ndarray::Array1;
use ndarray_rand::rand_distr::{Distribution, Exp};
use ndarray_rand::rand_distr::{Normal, Poisson};
use ndarray_rand::RandomExt;
use rand::thread_rng;

pub fn poisson(n: usize, lambda: usize, t_max: Option<f64>) -> Vec<f64> {
  if n == 0 || lambda == 0 {
    panic!("lambda, t and n must be positive integers");
  }

  let t_max = t_max.unwrap_or(1.0);
  let mut times = vec![0.0];
  let exp = Exp::new(lambda as f64).unwrap();

  while times.last().unwrap() < &t_max {
    let inter_arrival = exp.sample(&mut thread_rng());
    let next_time = times.last().unwrap() + inter_arrival;
    if next_time > t_max {
      break;
    }
    times.push(next_time);
  }

  times
}

pub fn compound_poisson(
  n: usize,
  lambda: f64,
  jumps: Option<Vec<f64>>,
  t_max: Option<f64>,
  jump_mean: Option<f64>,
  jump_std: Option<f64>,
) -> Vec<f64> {
  if n == 0 {
    panic!("n must be a positive integer");
  }

  let _t_max = t_max.unwrap_or(1.0);
  let p = Array1::random(n, Poisson::new(lambda).unwrap());
  let mut cp = Array1::<f64>::zeros(n);

  match jumps {
    Some(jumps) => jumps,
    None => {
      let _jump_mean = jump_mean.unwrap_or(0.0);
      let _jump_std = jump_std.unwrap_or(1.0);

      for i in 0..n {
        for j in &p {
          let norm = Array1::random(*j as usize, Normal::new(_jump_mean, _jump_std).unwrap());
          cp[i] = norm.sum();
        }
      }

      cp.to_vec()
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_poisson() {
    let n = 1000;
    let lambda = 10;
    let t = 10.0;
    let p = poisson(n, lambda, Some(t));
    println!("{:?}", p.len());
  }

  #[test]
  fn test_compound_poisson() {
    let n = 1000;
    let lambda = 2.0;
    let t = 10.0;
    let cp = compound_poisson(n, lambda, None, Some(t), None, None);
    assert_eq!(cp.len(), n);
  }
}
