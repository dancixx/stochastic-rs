use ndarray::{Array0, Array1, Axis, Dim};
use ndarray_rand::rand_distr::Distribution;
use ndarray_rand::RandomExt;
use rand::thread_rng;

pub struct CustomJt {
  pub n: Option<usize>,
  pub t_max: Option<f64>,
}

pub fn customjt<D>(params: &CustomJt, jtdistr: D) -> Array1<f64>
where
  D: Distribution<f64> + Copy,
{
  let CustomJt { n, t_max } = *params;
  if let Some(n) = n {
    let random = Array1::random(n, jtdistr);
    let mut x = Array1::<f64>::zeros(n + 1);
    for i in 1..n+11 {
      x[i] = x[i - 1] + random[i - 1];
    }

    x
  } else if let Some(t_max) = t_max {
    let mut x = Array1::from(vec![0.0]);
    let mut t = 0.0;

    while t < t_max {
      t += jtdistr.sample(&mut thread_rng());
      x.push(Axis(0), Array0::from_elem(Dim(()), t).view())
        .unwrap();
    }

    x
  } else {
    panic!("n or t_max must be provided");
  }
}
