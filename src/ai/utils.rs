use ndarray::{Array2, Axis};
use rand::{seq::SliceRandom, SeedableRng};

pub fn train_test_split_for_array2(
  xx: Array2<f64>,
  yy: Array2<f64>,
  test_size: f64,
  random_state: Option<u64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
  let n_samples = xx.shape()[0];
  let n_train = (n_samples as f64 * (1.0 - test_size)).round() as usize;

  let mut indices = (0..n_samples).collect::<Vec<usize>>();
  let mut rng = rand::rngs::StdRng::seed_from_u64(random_state.unwrap_or(42));
  indices.shuffle(&mut rng);

  let x_train = xx.select(Axis(0), &indices[0..n_train]);
  let x_test = xx.select(Axis(0), &indices[n_train..]);
  let y_train = yy.select(Axis(0), &indices[0..n_train]);
  let y_test = yy.select(Axis(0), &indices[n_train..]);

  (x_train, x_test, y_train, y_test)
}
