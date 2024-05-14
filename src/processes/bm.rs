use ndarray::{Array1, Axis};

use crate::noises::gn;

pub fn bm(n: usize, t: Option<f32>) -> Vec<f32> {
  let gn = gn::gn(n - 1, Some(t.unwrap_or(1.0)));
  let mut bm = Array1::<f32>::from(gn);
  bm.accumulate_axis_inplace(Axis(0), |&x, y| *y += x);
  vec![0.0].into_iter().chain(bm).collect()
}
