use std::time::Instant;

use plotly::{common::Line, Plot, Scatter};
use rand_distr::{Gamma, Normal};
use stochastic_rs::{
  jumps::{bates::bates_1996, jump_fou::jump_fou, levy_diffusion::levy_diffusion, merton::merton},
  processes::{cpoisson::compound_poisson, fbm::Fbm, poisson::poisson},
  quant::{diffusions::fbm, traits_f::SamplingF},
  utils::Generator,
};

fn main() {
  let mut plot = Plot::new();
  let fbm = fbm::FBM::new_f32(0.7, 100000, 0.0, 0.0, 1.0);
  let fbm2 = Fbm::new(0.7, 100000, Some(1.0), Some(10000));

  for i in 0..1 {
    // let d = poisson(10.0, Some(50), None);
    //   )
    //   .name("Poisson");
    //plot.add_trace(trace);
  }

  plot.show();

  //   // let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), d.clone())
  //   //   .mode(plotly::common::Mode::Lines)
  //   //   .line(
  //   //     Line::new().color("blue"), //.shape(plotly::common::LineShape::Hv),
  //   //   )
  //   //   .name(format!("Sequence {}", i + 1).as_str());
  //   // let trace = Scatter::new((0..d2.len()).collect::<Vec<_>>(), d2.clone())
  //   //   .mode(plotly::common::Mode::Lines)
  //   //   .line(
  //   //     Line::new().color("red"), //.shape(plotly::common::LineShape::Hv),
  //   //   )
  //   //   .name(format!("Sequence {}", i + 1).as_str());
  //   // plot.add_trace(trace);
  // }

  let start = Instant::now();
  // for i in 0..10000 {
  //   let d2 = fbm2.sample();
  // }
  fbm2.sample_par();
  println!("{}", start.elapsed().as_secs_f64());

  //plot.show();
}
