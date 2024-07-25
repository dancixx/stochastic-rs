use std::time::Instant;

use plotly::{common::Line, Plot, Scatter};
use rand_distr::{Gamma, Normal};
use stochastic_rs::{
  jumps::{bates::bates_1996, jump_fou::jump_fou, levy_diffusion::levy_diffusion, merton::merton},
  processes::{
    fbm::Fbm,
    poisson::{compound_poisson, poisson},
  },
  utils::Generator,
};

fn main() {
  let start = Instant::now();

  let mut plot = Plot::new();
  let fbm = Fbm::new(0.9, 1000, Some(1.0), Some(10));

  for i in 0..1 {
    let d = fbm.sample();
    let jump_fou = jump_fou(
      0.7,
      25.0,
      2.0,
      0.5,
      0.5,
      100,
      Some(0.0),
      Some(100.0),
      Normal::new(2.0, 3.5).unwrap(),
    );
    let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), jump_fou.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("blue")
          .shape(plotly::common::LineShape::Hv),
      )
      .name("Jump FOU");
    plot.add_trace(trace);
    let merton_path = merton(
      0.05,
      0.2,
      10.,
      1.0,
      1000,
      Some(1000.0),
      Some(5.0),
      Normal::new(0.0, 1.0).unwrap(),
    );
    let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), merton_path.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("red")
          .shape(plotly::common::LineShape::Hv),
      )
      .name("Merton");
    //plot.add_trace(trace);
    let levy_path = levy_diffusion(
      0.5,
      1.0,
      0.25,
      1000,
      Some(0.0),
      Some(10.0),
      Normal::new(0.0, 1.0).unwrap(),
    );
    let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), levy_path.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("green")
          .shape(plotly::common::LineShape::Hv),
      )
      .name("Levy");
    //plot.add_trace(trace);
    let [s, v] = bates_1996(
      0.05,
      1.5,
      0.04,
      0.3,
      -0.7,
      0.1,
      1000,
      Some(100.0),
      Some(0.04),
      Some(1.0),
      Some(false),
      Normal::new(0.0, 1.0).unwrap(),
    );

    let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), s.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("black")
          .shape(plotly::common::LineShape::Linear),
      )
      .name("Bates 1996");
    //plot.add_trace(trace);

    let [_, cp, _] = compound_poisson(None, 1.0, Some(500.), Normal::new(0.0, 1.0).unwrap());
    let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), cp.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("purple")
          .shape(plotly::common::LineShape::Hv),
      )
      .name("Compound Poisson");
    //plot.add_trace(trace);
    let d = poisson(10.0, Some(50), None);
    let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), d.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("orange")
          .shape(plotly::common::LineShape::Hv),
      )
      .name("Poisson");
    //plot.add_trace(trace);
  }

  plot.show();

  println!("{}", start.elapsed().as_secs_f64());
}
