use std::time::Instant;

use plotly::{
  common::{Line, TickMode},
  Plot, Scatter,
};
use stochastic_rs::{
  processes::{
    fbm::Fbm,
    poisson::{compound_poisson, poisson},
  },
  quant::{diffusions::fbm, traits_f::SamplingF},
  utils::Generator,
};

fn main() {
  let mut plot = Plot::new();
  let fbm = fbm::FBM::new_f32(0.7, 100000, 0.0, 0.0, 1.0);
  let fbm2 = Fbm::new(0.7, 100000, Some(1.0), Some(10000));

  let start = Instant::now();
  fbm.sample_parallel(10000);
  // for i in 0..10000 {
  //   let d = fbm.sample();

  //   // let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), d.clone())
  //   //   .mode(plotly::common::Mode::Lines)
  //   //   .line(
  //   //     Line::new().color("blue"), //.shape(plotly::common::LineShape::Hv),
  //   //   )
  //   //   .name(format!("Sequence {}", i + 1).as_str());
  //   // plot.add_trace(trace);
  //   // let trace = Scatter::new((0..d2.len()).collect::<Vec<_>>(), d2.clone())
  //   //   .mode(plotly::common::Mode::Lines)
  //   //   .line(
  //   //     Line::new().color("red"), //.shape(plotly::common::LineShape::Hv),
  //   //   )
  //   //   .name(format!("Sequence {}", i + 1).as_str());
  //   // plot.add_trace(trace);
  // }
  println!("{}", start.elapsed().as_secs_f64());

  let start = Instant::now();
  // for i in 0..10000 {
  //   let d2 = fbm2.sample();
  // }
  fbm2.sample_par();
  println!("{}", start.elapsed().as_secs_f64());

  //plot.show();
}
