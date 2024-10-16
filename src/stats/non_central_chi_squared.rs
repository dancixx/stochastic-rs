use rand_distr::{ChiSquared, Distribution, Normal};

pub fn sample(df: f64, non_centrality: f64) -> f64 {
  let mut rng = rand::thread_rng();

  let chi_squared = ChiSquared::new(df).unwrap();
  let central_part: f64 = chi_squared.sample(&mut rng);

  let normal_dist = Normal::new(non_centrality.sqrt(), 1.0).unwrap();
  let non_central_part: f64 = normal_dist.sample(&mut rng).powi(2);

  central_part + non_central_part
}
