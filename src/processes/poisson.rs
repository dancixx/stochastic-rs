use ndarray::{Array0, Array1, Axis, Dim};
use ndarray_rand::rand_distr::{Distribution, Exp};
use ndarray_rand::rand_distr::{Normal, Poisson};
use ndarray_rand::RandomExt;
use rand::thread_rng;

pub fn poisson(lambda: f32, n: Option<usize>, t_max: Option<f32>) -> Vec<f32> {
  if let Some(n) = n {
    let exponentials = Array1::random(n - 1, Exp::new(lambda).unwrap());
    let mut poisson = Array1::<f32>::zeros(n);

    for i in 1..n {
      poisson[i] = poisson[i - 1] + exponentials[i - 1];
    }

    poisson.to_vec()
  } else if let Some(t_max) = t_max {
    let mut poisson = Array1::from(vec![0.0]);
    let mut t = 0.0;

    while &t < &t_max {
      t += Exp::new(lambda).unwrap().sample(&mut thread_rng());
      poisson
        .push(Axis(0), Array0::from_elem(Dim(()), t).view())
        .unwrap();
    }

    poisson.to_vec()
  } else {
    panic!("n or t_max must be provided");
  }
}

pub fn compound_poisson(
  n: usize,
  lambda: f32,
  jumps: Option<Vec<f32>>,
  t_max: Option<f32>,
  jump_mean: Option<f32>,
  jump_std: Option<f32>,
) -> Vec<f32> {
  if n == 0 {
    panic!("n must be a positive integer");
  }

  let _t_max = t_max.unwrap_or(1.0);
  let p = Array1::random(n, Poisson::new(lambda).unwrap());
  let mut cp = Array1::<f32>::zeros(n);

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
    let lambda = 1;
    let p = poisson(lambda as f32, Some(n), None);
    println!("{:?}", p.len());
    let t = 100.0;
    let p = poisson(lambda as f32, None, Some(t));
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
