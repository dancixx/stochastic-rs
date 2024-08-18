use derive_builder::Builder;
use ndarray::{Array1, Axis};
use rand::thread_rng;
use rand_distr::Distribution;

use super::customjt::{customjt, CustomJt};

#[derive(Default, Builder)]
#[builder(setter(into))]
pub struct CompoundCustom {
  pub n: Option<usize>,
  pub t_max: Option<f64>,
}

pub fn compound_custom<D, E>(params: &CompoundCustom, jdistr: D, jtdistr: E) -> [Array1<f64>; 3]
where
  D: Distribution<f64> + Copy,
  E: Distribution<f64> + Copy,
{
  let CompoundCustom { n, t_max } = *params;
  if n.is_none() && t_max.is_none() {
    panic!("n or t_max must be provided");
  }

  let p = customjt(&CustomJt { n, t_max }, jtdistr);
  let mut jumps = Array1::<f64>::zeros(n.unwrap_or(p.len()));
  for i in 1..p.len() {
    jumps[i] = jdistr.sample(&mut thread_rng());
  }

  let mut cum_jupms = jumps.clone();
  cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

  [p, cum_jupms, jumps]
}
