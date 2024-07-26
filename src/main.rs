use plotly::{common::Line, Layout, Plot};
use rand_distr::{Exp, Normal};
use stochastic_rs::jumps::{
  bates::{bates_1996, Bates1996},
  jump_fou::{jump_fou, JumpFou},
  levy_diffusion::{levy_diffusion, LevyDiffusion},
  merton::{merton, Merton},
};

fn main() {
  let mut plot = Plot::new();
  plot.set_layout(Layout::new().width(600).height(600));

  for _ in 0..1 {
    // let jump_fou = jump_fou(
    //   &JumpFou {
    //     hurst: 0.75,
    //     mu: 10.0,
    //     sigma: 9.0,
    //     theta: 1.0,
    //     n: 100,
    //     t: Some(1.0),
    //     lambda: 0.25,
    //     ..Default::default()
    //   },
    //   Exp::new(1.0).unwrap(),
    // );
    // let trace = plotly::Scatter::new(
    //   (0..jump_fou.len())
    //     .into_iter()
    //     .map(|idx| idx)
    //     .collect::<Vec<_>>(),
    //   jump_fou.to_vec(),
    // )
    // .line(
    //   Line::new()
    //     .color("blue")
    //     .shape(plotly::common::LineShape::Hv),
    // );
    // plot.add_trace(trace);

    // let jump_fou = levy_diffusion(
    //   &LevyDiffusion {
    //     gamma: 1.0,
    //     sigma: 0.25,
    //     n: 100,
    //     t: Some(10.0),
    //     lambda: 40.0,
    //     ..Default::default()
    //   },
    //   Normal::new(0.25, 2.0).unwrap(),
    // );
    // let trace = plotly::Scatter::new(
    //   (0..jump_fou.len())
    //     .into_iter()
    //     .map(|idx| idx)
    //     .collect::<Vec<_>>(),
    //   jump_fou.to_vec(),
    // )
    // .line(
    //   Line::new()
    //     .color("blue")
    //     .shape(plotly::common::LineShape::Hv),
    // );
    // plot.add_trace(trace);

    // let merton: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> = merton(
    //   &Merton {
    //     alpha: 0.08,
    //     sigma: 0.1,
    //     lambda: 25.0,
    //     theta: 0.08,
    //     n: 100,
    //     x0: Some(40.0),
    //     t: Some(1.0),
    //   },
    //   Normal::new(0.0, 2.0).unwrap(),
    // );
    // let trace = plotly::Scatter::new(
    //   (0..merton.len())
    //     .into_iter()
    //     .map(|idx| idx)
    //     .collect::<Vec<_>>(),
    //   merton.to_vec(),
    // )
    // .line(
    //   Line::new()
    //     .color("blue")
    //     .shape(plotly::common::LineShape::Hv),
    // );
    // plot.add_trace(trace);

    let [s, v] = bates_1996(
      &Bates1996 {
        mu: 1.0,
        kappa: 1.0,
        theta: 1.0,
        eta: 1.0,
        rho: 0.03,
        lambda: 2.0,
        n: 1000,
        s0: Some(0.0),
        v0: Some(0.0),
        t: Some(1.0),
        use_sym: Some(false),
      },
      Normal::new(0.0, 1.0).unwrap(),
    );
    let trace = plotly::Scatter::new(
      (0..s.len()).into_iter().map(|idx| idx).collect::<Vec<_>>(),
      s.to_vec(),
    )
    .line(
      Line::new()
        .color("blue")
        .shape(plotly::common::LineShape::Hv),
    );
    plot.add_trace(trace);
  }

  plot.show();
}
