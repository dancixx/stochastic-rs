use stochastic_rs::{processes::fbm::Fbm, utils::Generator};

fn main() {
  let fbm = Fbm::new(0.75, 10000, Some(1.0), None);
  let mut runs = Vec::new();

  for _ in 0..20 {
    let start = std::time::Instant::now();
    for _ in 0..1000 {
      let _ = fbm.sample();
    }

    let duration = start.elapsed();
    println!(
      "Time elapsed in expensive_function() is: {:?}",
      duration.as_secs_f32()
    );
    runs.push(duration.as_secs_f32());
  }

  let sum: f32 = runs.iter().sum();
  let average = sum / runs.len() as f32;
  println!("Average time: {}", average);

  let start = std::time::Instant::now();
  let fbm = Fbm::new(0.75, 10000, Some(1.0), Some(1000));
  let data = fbm.sample_par();
  println!("Data: {:?}", data);
  let duration = start.elapsed();
  println!(
    "Time elapsed in expensive_function() is: {:?}",
    duration.as_secs_f32()
  );
}
