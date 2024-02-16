use ndarray::Array1;
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Exp, Normal};

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
  lambda: usize,
  jumps: Option<Vec<f64>>,
  t_max: Option<f64>,
  jump_mean: Option<f64>,
  jump_std: Option<f64>,
) -> [Vec<f64>; 2] {
  if n == 0 {
    panic!("n must be a positive integer");
  }

  let _t_max = t_max.unwrap_or(1.0);
  let times = poisson(n, lambda, Some(_t_max));
  let mut cp = Array1::<f64>::zeros(times.len());
  let jumps = match jumps {
    Some(jumps) => jumps,
    None => {
      let _jump_mean = jump_mean.unwrap_or(1.0);
      let _jump_std = jump_std.unwrap_or(0.0);
      let norm = Normal::new(_jump_mean, _jump_std).unwrap();
      thread_rng().sample_iter(norm).take(n).collect()
    }
  };

  for i in 1..times.len() {
    cp[i] = cp[i - 1] + jumps[i - 1];
  }

  [times, cp.to_vec()]
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
    let lambda = 10;
    let t = 10.0;
    let cp = compound_poisson(n, lambda, None, Some(t), Some(1.0), Some(0.0));
    println!("{:?}", cp[0].len());
    println!("{:?}", cp[1].len());
  }
}
