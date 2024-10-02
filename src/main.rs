use stochastic_rs::stochastic::{noise::fgn::FGN, Sampling};

fn main() {
  let fbm = FGN::new(0.9, 10000, None, Some(10000));

  let start = std::time::Instant::now();
  for _ in 0..10000 {
    let _ = fbm.sample();
  }

  let duration = start.elapsed();
  println!(
    "Time elapsed in expensive_function() is: {:?}",
    duration.as_secs_f32()
  );

  let fbm = FGN::new(0.9, 10000, None, Some(10000));

  let start = std::time::Instant::now();
  for _ in 0..10000 {
    let _ = fbm.sample();
  }

  let duration = start.elapsed();
  println!(
    "Time elapsed in expensive_function() is: {:?}",
    duration.as_secs_f32()
  );
}
